using System;
using System.Security.Cryptography;
using NumSharp.Backends;
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
        ///     NumPy 2.4.2's bits for complex128 np.correlate / np.convolve through the per-position engine
        ///     (<c>SlidingDot.SlidingComplex</c>: CDOUBLE_dot per output position over zdotu - its four sums below 8 terms,
        ///     its vector kernel from 8 - which NumPy runs at every position of a complex correlation). The blocked long
        ///     kernel is turned off on this thread (the route a kernel holding a non-finite value takes, and a host without
        ///     AVX / FMA); a kernel shorter than 8 terms takes this engine anyway.
        /// </summary>
        /// <param name="fn">correlate or convolve.</param>
        /// <param name="mode">full, same or valid.</param>
        /// <param name="n1">The data length.</param>
        /// <param name="n2">The kernel length.</param>
        /// <param name="seed">The generator seed (the data drawn first, then the kernel).</param>
        /// <param name="expected">sha: and the first 32 hex digits of SHA-256 over NumPy's result bytes.</param>
        [DataTestMethod]
        [DataRow("correlate", "full", 3, 3, 9001UL, "sha:e795c8a78f51ddd4b206e24af5ffca5c")]
        [DataRow("correlate", "same", 3, 3, 9002UL, "sha:0932701eac6320ca2551af5ceb799f10")]
        [DataRow("correlate", "valid", 3, 3, 9003UL, "sha:fec204e07ec53b97b3f638ccad96619e")]
        [DataRow("convolve", "full", 3, 3, 9004UL, "sha:d997fda41291e6fea4dae6d3608ab4d6")]
        [DataRow("convolve", "same", 3, 3, 9005UL, "sha:9c60286b6fb8d15b9579e3e933c682d0")]
        [DataRow("convolve", "valid", 3, 3, 9006UL, "sha:39ed09d932806567fc8f1e949ff50f53")]
        [DataRow("correlate", "full", 8, 3, 9007UL, "sha:7da45d7d415f92d9b983bb89fbd9dd5b")]
        [DataRow("correlate", "same", 8, 3, 9008UL, "sha:522dbb48c041153ab707aaf1afe85d87")]
        [DataRow("correlate", "valid", 8, 3, 9009UL, "sha:5191c156ec944d827ee197750686d39f")]
        [DataRow("convolve", "full", 8, 3, 9010UL, "sha:024701a8cfb09cc5cd988f4756f0720e")]
        [DataRow("convolve", "same", 8, 3, 9011UL, "sha:f54b55893a9e5ddd5f8000a9bb506493")]
        [DataRow("convolve", "valid", 8, 3, 9012UL, "sha:d8b3034bf78fc2a6c1e3fff5eee2ff67")]
        [DataRow("correlate", "full", 40, 3, 9013UL, "sha:0fa0b840a3f93475d130d7611953e941")]
        [DataRow("correlate", "same", 40, 3, 9014UL, "sha:18bc50168a6e7d1e44140fec288014d6")]
        [DataRow("correlate", "valid", 40, 3, 9015UL, "sha:8853eaba4a5caeb82e1a36b0e272900a")]
        [DataRow("convolve", "full", 40, 3, 9016UL, "sha:df609df7aa97cab738854ea50867c139")]
        [DataRow("convolve", "same", 40, 3, 9017UL, "sha:9506efe3d8700ff26c61d7349dae62ac")]
        [DataRow("convolve", "valid", 40, 3, 9018UL, "sha:65146bb812c2e024261a8607f9f55a56")]
        [DataRow("correlate", "full", 7, 7, 9019UL, "sha:b4fe45a2a4dcbfdf7370addcc6b10812")]
        [DataRow("correlate", "same", 7, 7, 9020UL, "sha:529c96a2fe5da325c3ff6623678963a6")]
        [DataRow("correlate", "valid", 7, 7, 9021UL, "sha:a94a79fb46d13bf7cd7d6157e9c9d43b")]
        [DataRow("convolve", "full", 7, 7, 9022UL, "sha:41ffc6b8eb2541aa152de31cdd8b1f3d")]
        [DataRow("convolve", "same", 7, 7, 9023UL, "sha:ac3a4f32959710797a54fc0e365f4680")]
        [DataRow("convolve", "valid", 7, 7, 9024UL, "sha:6eceb7aa9c7c117aac9e27729b539a0f")]
        [DataRow("correlate", "full", 12, 7, 9025UL, "sha:6e07fed74e7bedb935c0a326861251a7")]
        [DataRow("correlate", "same", 12, 7, 9026UL, "sha:67f77c0587b53bd42c29e6e797d03055")]
        [DataRow("correlate", "valid", 12, 7, 9027UL, "sha:c8b8f70e009defff886737f7674d7872")]
        [DataRow("convolve", "full", 12, 7, 9028UL, "sha:7a725a6b7a76deb0318c9eb2d352b7cd")]
        [DataRow("convolve", "same", 12, 7, 9029UL, "sha:99cea24693578cd765a62a0b503dc0a9")]
        [DataRow("convolve", "valid", 12, 7, 9030UL, "sha:281ae2ac2f585d30eaac74d565351593")]
        [DataRow("correlate", "full", 40, 7, 9031UL, "sha:9f12079fe05e4250c141e04bc4dcebb0")]
        [DataRow("correlate", "same", 40, 7, 9032UL, "sha:07ff6c16ddccf387af1603d23dc22028")]
        [DataRow("correlate", "valid", 40, 7, 9033UL, "sha:4d6421290b49e23602150b0c859d2d62")]
        [DataRow("convolve", "full", 40, 7, 9034UL, "sha:193222a803d1bfc964dab2c6bf64af2c")]
        [DataRow("convolve", "same", 40, 7, 9035UL, "sha:b9a48f178500e8b1da221bcbb8308dc3")]
        [DataRow("convolve", "valid", 40, 7, 9036UL, "sha:f59c5228d9039e1160001942eb51886c")]
        [DataRow("correlate", "full", 8, 8, 9037UL, "sha:b66e74225cb71f5d6b5a8543f9a725dd")]
        [DataRow("correlate", "same", 8, 8, 9038UL, "sha:56c3998c765eb0ec48325dc576f475a4")]
        [DataRow("correlate", "valid", 8, 8, 9039UL, "sha:2851df194b5456505d9eb7fda3c1820c")]
        [DataRow("convolve", "full", 8, 8, 9040UL, "sha:30abe3fac43b057572cb24a1b67b2505")]
        [DataRow("convolve", "same", 8, 8, 9041UL, "sha:cca4ff4c4fb93b5935549043924469ee")]
        [DataRow("convolve", "valid", 8, 8, 9042UL, "sha:4af761da3df0bb73b8bc2eabc5b9fcb8")]
        [DataRow("correlate", "full", 13, 8, 9043UL, "sha:b53c198edea8b7777ba635b34e11e700")]
        [DataRow("correlate", "same", 13, 8, 9044UL, "sha:3401913ac30e2fa04b73b4f363332b3d")]
        [DataRow("correlate", "valid", 13, 8, 9045UL, "sha:eab309cc5fe77cd9ac433b9a8b9e6630")]
        [DataRow("convolve", "full", 13, 8, 9046UL, "sha:5ceffd52bee44f6206d36eab798430b8")]
        [DataRow("convolve", "same", 13, 8, 9047UL, "sha:0d062c5acf5b84742dccba8e0753e807")]
        [DataRow("convolve", "valid", 13, 8, 9048UL, "sha:caf8f8a4dd965634980319d69e0fd437")]
        [DataRow("correlate", "full", 40, 8, 9049UL, "sha:54bf713fe125ad40692794ccea467754")]
        [DataRow("correlate", "same", 40, 8, 9050UL, "sha:df4b34639c32f56d0fdcb59bd5287c47")]
        [DataRow("correlate", "valid", 40, 8, 9051UL, "sha:030d3aa3e5910bc1cf1df12cfb51c8e1")]
        [DataRow("convolve", "full", 40, 8, 9052UL, "sha:6aba70405ecb4da36df0240b55d5ffb3")]
        [DataRow("convolve", "same", 40, 8, 9053UL, "sha:ffe0ac21bf1ffb48a58417419ddb15a4")]
        [DataRow("convolve", "valid", 40, 8, 9054UL, "sha:7ec1ea55f517562d6f30825f2d81c9b7")]
        [DataRow("correlate", "full", 9, 9, 9055UL, "sha:a80a7f0043836d077878ea7073d62d05")]
        [DataRow("correlate", "same", 9, 9, 9056UL, "sha:4409ac562c9cb55ae2313f8622d59130")]
        [DataRow("correlate", "valid", 9, 9, 9057UL, "sha:68ef6fa65932522b619a169ef85a11e3")]
        [DataRow("convolve", "full", 9, 9, 9058UL, "sha:93e120053ff8218cff01b9396a6e2f24")]
        [DataRow("convolve", "same", 9, 9, 9059UL, "sha:1213600523e05b533d8ce7768cca38b4")]
        [DataRow("convolve", "valid", 9, 9, 9060UL, "sha:ce8997b06467ada4cbc8a3fcdf98a8df")]
        [DataRow("correlate", "full", 14, 9, 9061UL, "sha:453a2b869a07cb54c8029606c9b64191")]
        [DataRow("correlate", "same", 14, 9, 9062UL, "sha:e646cb8d615b387e3a83161e1bd5ed7e")]
        [DataRow("correlate", "valid", 14, 9, 9063UL, "sha:b0e2e54ba73358fb6f1cd6f288b23c48")]
        [DataRow("convolve", "full", 14, 9, 9064UL, "sha:956fc54d89185d37bf84bbb13fc9d271")]
        [DataRow("convolve", "same", 14, 9, 9065UL, "sha:1780b437f5301c004ff4b7c71737cace")]
        [DataRow("convolve", "valid", 14, 9, 9066UL, "sha:7567e7df6bfaa208e15259695a006c60")]
        [DataRow("correlate", "full", 40, 9, 9067UL, "sha:4fd412bebfa19e92af69c360baf9b3b8")]
        [DataRow("correlate", "same", 40, 9, 9068UL, "sha:27e87f2c08931d4af1657921376d5e2b")]
        [DataRow("correlate", "valid", 40, 9, 9069UL, "sha:24acf00dd93be3e4fdd1ecc1c1d7a7c7")]
        [DataRow("convolve", "full", 40, 9, 9070UL, "sha:69de845c380a8b02c15c62c253e79dba")]
        [DataRow("convolve", "same", 40, 9, 9071UL, "sha:6c8561ec5e603833decf1599dced6136")]
        [DataRow("convolve", "valid", 40, 9, 9072UL, "sha:c2e4d33044a7d1ab8e3facc8e77403cf")]
        [DataRow("correlate", "full", 15, 15, 9073UL, "sha:4cb9bcb84aa8a09a39dcb12c0ee8f9bd")]
        [DataRow("correlate", "same", 15, 15, 9074UL, "sha:f7075aa5d836bb720e44b6a25deadd98")]
        [DataRow("correlate", "valid", 15, 15, 9075UL, "sha:5fc0cd922740a633ab2b5d11b9849aa3")]
        [DataRow("convolve", "full", 15, 15, 9076UL, "sha:238fa11dc769205ed14ca1b032dc7f14")]
        [DataRow("convolve", "same", 15, 15, 9077UL, "sha:4ed8904619152d00d46d7e52e005f619")]
        [DataRow("convolve", "valid", 15, 15, 9078UL, "sha:b7aead636e2159dc914672a61a603e86")]
        [DataRow("correlate", "full", 20, 15, 9079UL, "sha:685a3ae20ae9435de8f1f0ce35e1bbd4")]
        [DataRow("correlate", "same", 20, 15, 9080UL, "sha:87318ed482be0eee27b907c7f16b5081")]
        [DataRow("correlate", "valid", 20, 15, 9081UL, "sha:9a7edc45c62288c07af1165891492ba4")]
        [DataRow("convolve", "full", 20, 15, 9082UL, "sha:0577f84ea46670c6ee58876c3d29d15d")]
        [DataRow("convolve", "same", 20, 15, 9083UL, "sha:f110e94bf5934baa4b74253cf8ce8a04")]
        [DataRow("convolve", "valid", 20, 15, 9084UL, "sha:b04c6b02dca6b0e716975ad1ff938586")]
        [DataRow("correlate", "full", 40, 15, 9085UL, "sha:6faaf7775ce9eff9bc6d4b28af86f56e")]
        [DataRow("correlate", "same", 40, 15, 9086UL, "sha:458dc5d3579e1b961b7776e51df638d3")]
        [DataRow("correlate", "valid", 40, 15, 9087UL, "sha:3790c884b8666b8c29f68291cb03d6c3")]
        [DataRow("convolve", "full", 40, 15, 9088UL, "sha:427423bd0969db270f8c0e8ad67a232a")]
        [DataRow("convolve", "same", 40, 15, 9089UL, "sha:83b41168b4c407c83dcd972eb2797560")]
        [DataRow("convolve", "valid", 40, 15, 9090UL, "sha:b103db32e6c63ec4514dab0dca6b23f4")]
        [DataRow("correlate", "full", 16, 16, 9091UL, "sha:1455daddfba8a5c90a3e90cb42dcc4c3")]
        [DataRow("correlate", "same", 16, 16, 9092UL, "sha:428966119c85409b93a5f8e499e5ec2a")]
        [DataRow("correlate", "valid", 16, 16, 9093UL, "sha:d9566ecb099ff15fcd39363b4757e1df")]
        [DataRow("convolve", "full", 16, 16, 9094UL, "sha:41534db6378ab32a9bc7d44399a60aa9")]
        [DataRow("convolve", "same", 16, 16, 9095UL, "sha:074511317f459a0f089dbc493a8b194b")]
        [DataRow("convolve", "valid", 16, 16, 9096UL, "sha:b70984360487e16f8d2f552217f74a49")]
        [DataRow("correlate", "full", 21, 16, 9097UL, "sha:15dc70ca3eca6db1bc498c416deaeb73")]
        [DataRow("correlate", "same", 21, 16, 9098UL, "sha:9d479998e08c18dbff9a78e4703a2fb5")]
        [DataRow("correlate", "valid", 21, 16, 9099UL, "sha:660ee57817c2f327294b8fa7504f4206")]
        [DataRow("convolve", "full", 21, 16, 9100UL, "sha:c075b0d75750f14f85242839175ea955")]
        [DataRow("convolve", "same", 21, 16, 9101UL, "sha:3318c08bd88925b0106e67609c73a497")]
        [DataRow("convolve", "valid", 21, 16, 9102UL, "sha:17faae9cbf5bfe4db3ce0e9ddfcaa9bb")]
        [DataRow("correlate", "full", 40, 16, 9103UL, "sha:a4c18bc4efd3c6d760ce5c89f5a91010")]
        [DataRow("correlate", "same", 40, 16, 9104UL, "sha:8af86c4f425e83b267b775a71daae5ae")]
        [DataRow("correlate", "valid", 40, 16, 9105UL, "sha:47cfdc06c8f68f3dc58acf3493b8d65c")]
        [DataRow("convolve", "full", 40, 16, 9106UL, "sha:5418cdd44354875e64953aa6a5ceb9de")]
        [DataRow("convolve", "same", 40, 16, 9107UL, "sha:af489cabc7bcc6a4e49188acf0499fe9")]
        [DataRow("convolve", "valid", 40, 16, 9108UL, "sha:dd0cdab1ac9113cb12e431a75ef8e88c")]
        [DataRow("correlate", "full", 17, 17, 9109UL, "sha:e6b5c928d9b54e02afd1ba032be4db84")]
        [DataRow("correlate", "same", 17, 17, 9110UL, "sha:eb5a08fc318b23728c014e49b0ee2c72")]
        [DataRow("correlate", "valid", 17, 17, 9111UL, "sha:8cf1f25e98797be690bcd2f078780c7e")]
        [DataRow("convolve", "full", 17, 17, 9112UL, "sha:7fb85ca4beee84d7be9f9836a7744bb7")]
        [DataRow("convolve", "same", 17, 17, 9113UL, "sha:53b407ae298607745a73d5f009a329c9")]
        [DataRow("convolve", "valid", 17, 17, 9114UL, "sha:5205b930b525c9f0f743be7f59db2d5b")]
        [DataRow("correlate", "full", 22, 17, 9115UL, "sha:f448a36738ff0dfd79e8133346099e9a")]
        [DataRow("correlate", "same", 22, 17, 9116UL, "sha:48219e904f22f50a80a3baf8cf596e64")]
        [DataRow("correlate", "valid", 22, 17, 9117UL, "sha:746e218da9f9b54e3fb1d5c2188ae919")]
        [DataRow("convolve", "full", 22, 17, 9118UL, "sha:fd507b0bfbadf569529ce7d993b08c59")]
        [DataRow("convolve", "same", 22, 17, 9119UL, "sha:d32c8986ad3cea573602eb04a9218e0c")]
        [DataRow("convolve", "valid", 22, 17, 9120UL, "sha:7b86cf367491adf07b5b7b345fe6d7e1")]
        [DataRow("correlate", "full", 40, 17, 9121UL, "sha:cae21a1cef99b1951398194084dc5a7d")]
        [DataRow("correlate", "same", 40, 17, 9122UL, "sha:da5de2fddd7fa70c8942aa264365e74f")]
        [DataRow("correlate", "valid", 40, 17, 9123UL, "sha:5bb8843bb1e88f62723d0d04db7998f8")]
        [DataRow("convolve", "full", 40, 17, 9124UL, "sha:69a0c992650ae2ce1fffedee730fcfd6")]
        [DataRow("convolve", "same", 40, 17, 9125UL, "sha:3b53892e25d35e62588c2a74ce670df9")]
        [DataRow("convolve", "valid", 40, 17, 9126UL, "sha:d10fbdf18e9066efe363077244eec8f3")]
        [DataRow("correlate", "full", 31, 31, 9127UL, "sha:f00905b1cffb292956886a9824dd2476")]
        [DataRow("correlate", "same", 31, 31, 9128UL, "sha:e3c8409b1d3b36a6465d473ab145c5bc")]
        [DataRow("correlate", "valid", 31, 31, 9129UL, "sha:435a23e814e117e19a7b1721df690e93")]
        [DataRow("convolve", "full", 31, 31, 9130UL, "sha:b434dfaa6ecfec0684f37ab2cefdbeb9")]
        [DataRow("convolve", "same", 31, 31, 9131UL, "sha:ffe8d555f3c4aacb302c4bb5f2920e07")]
        [DataRow("convolve", "valid", 31, 31, 9132UL, "sha:afefea4f041b85326e8c40a1c75f628f")]
        [DataRow("correlate", "full", 36, 31, 9133UL, "sha:e47037694cb2b09adaa510202f1e9294")]
        [DataRow("correlate", "same", 36, 31, 9134UL, "sha:40a10a6e743388035be2eecd9af84100")]
        [DataRow("correlate", "valid", 36, 31, 9135UL, "sha:39d9d9760693a018a1f9579f22b24354")]
        [DataRow("convolve", "full", 36, 31, 9136UL, "sha:5773db787bc7c9ddbacf5008ede98c8f")]
        [DataRow("convolve", "same", 36, 31, 9137UL, "sha:35f78c8bbe612a58411410740fbf2f7a")]
        [DataRow("convolve", "valid", 36, 31, 9138UL, "sha:e9283adcc68f021576fdd30d86c54df5")]
        [DataRow("correlate", "full", 40, 31, 9139UL, "sha:2059d7cf5a2454db79858b9ab770bd40")]
        [DataRow("correlate", "same", 40, 31, 9140UL, "sha:f3cf0250f55deb21952dfde107fc673a")]
        [DataRow("correlate", "valid", 40, 31, 9141UL, "sha:6edee0c31be7d1c5bf1ce6b9d529c5f5")]
        [DataRow("convolve", "full", 40, 31, 9142UL, "sha:71607fa6757e81c761635b06651db39b")]
        [DataRow("convolve", "same", 40, 31, 9143UL, "sha:2d8b5a1d1bb2180a20fcf8d34d3d960a")]
        [DataRow("convolve", "valid", 40, 31, 9144UL, "sha:f0f4700e440dcc25569eb7ea7094a032")]
        [DataRow("correlate", "full", 33, 33, 9145UL, "sha:02d5b37502d34b3ff2f095d1172b4335")]
        [DataRow("correlate", "same", 33, 33, 9146UL, "sha:40ead138c80e913b3e29a4e09e5b3a48")]
        [DataRow("correlate", "valid", 33, 33, 9147UL, "sha:6a8e42fafb60633c75935f6f460bf1f6")]
        [DataRow("convolve", "full", 33, 33, 9148UL, "sha:b7a8c8969a40fb78947f4fb5adea79a5")]
        [DataRow("convolve", "same", 33, 33, 9149UL, "sha:2fe9337e1a07114524dd7d2a91a0ab0c")]
        [DataRow("convolve", "valid", 33, 33, 9150UL, "sha:82186270d4f073a7f704a935b36d016c")]
        [DataRow("correlate", "full", 38, 33, 9151UL, "sha:5e7a868e1397692dede72527b766ae51")]
        [DataRow("correlate", "same", 38, 33, 9152UL, "sha:68017c6e14e69817525c3b13217873a4")]
        [DataRow("correlate", "valid", 38, 33, 9153UL, "sha:e3cface9f80bf47359dfa0c57157c9c4")]
        [DataRow("convolve", "full", 38, 33, 9154UL, "sha:a39cdacf7375ce994665af08be199aa6")]
        [DataRow("convolve", "same", 38, 33, 9155UL, "sha:170e428c2173b8e1b06839ba14ec7490")]
        [DataRow("convolve", "valid", 38, 33, 9156UL, "sha:c29c9c013405f3cc94be93f2f7b122ab")]
        [DataRow("correlate", "full", 40, 33, 9157UL, "sha:f8ecaed62c165f8fd00bff68784aac85")]
        [DataRow("correlate", "same", 40, 33, 9158UL, "sha:a3499fa4820b916f7dd9e7edf0545e6f")]
        [DataRow("correlate", "valid", 40, 33, 9159UL, "sha:4514e3e283dc77a847787d1bc80d9de6")]
        [DataRow("convolve", "full", 40, 33, 9160UL, "sha:c517d76c33beef81856d252d4f753200")]
        [DataRow("convolve", "same", 40, 33, 9161UL, "sha:8c391a3abe7073e9286b14ad7be52a61")]
        [DataRow("convolve", "valid", 40, 33, 9162UL, "sha:ba674b411f803f7d6a29896661fbc833")]
        public void ComplexSliding_PerPosition_MatchesNumPyBits(string fn, string mode, int n1, int n2, ulong seed,
            string expected)
        {
            var g = new Lcg(seed);
            var a = g.Array(new long[] { n1 }, "complex128");
            var v = g.Array(new long[] { n2 }, "complex128");
            // The hook is [ThreadStatic]: it changes this test's thread only, and the finally restores it for the next
            // test the worker runs.
            SlidingDot.LongKernelsDisabled = true;
            try
            {
                var r = fn == "correlate" ? np.correlate(a, v, mode) : np.convolve(a, v, mode);
                AssertNumPyBits(r, "complex128", expected, $"{fn} {mode} n1={n1} n2={n2}");
            }
            finally
            {
                SlidingDot.LongKernelsDisabled = false;
            }
        }

        /// <summary>
        ///     NumPy 2.4.2's bits for np.vecdot (FLOAT_vecdot / DOUBLE_vecdot / CDOUBLE_vecdot: the loop dtype's dot
        ///     function per output, CDOUBLE_dotc conjugating the first operand) over every route the engine takes: each
        ///     core stride class (unit, the strided loops, the own loop with its four-output pair rows), an operand
        ///     broadcast over the loop axes or along its core, an operand cast to the loop dtype (read at unit stride, as
        ///     NumPy's iterator copy is, a block of runs at a time), and a core of one element (the own loop's single term,
        ///     vectorized over 37 outputs).
        /// </summary>
        /// <param name="dtypeA">The first operand's dtype.</param>
        /// <param name="dtypeB">The second operand's dtype.</param>
        /// <param name="kind">c, rev (<c>a[:, ::-1]</c>), b2 (b of shape (n,)), t1 (<c>a.T</c>), tt (both transposed), step
        ///     (both <c>[:, ::2]</c>), bcore (b broadcast along the core), d3 (loop axes (2,1,3) and (4,1)).</param>
        /// <param name="n">The core length.</param>
        /// <param name="m">The number of outputs along the loop axis (the 2-D kinds).</param>
        /// <param name="seed">The generator seed (a drawn first, then b).</param>
        /// <param name="special">Whether NaN / -0.0 / inf / -inf are planted (then compared byte for byte, NaN slots as
        ///     NaN).</param>
        /// <param name="resultDtype">NumPy's result dtype (the loop dtype).</param>
        /// <param name="expected">sha: or hex: and NumPy's result bytes.</param>
        [DataTestMethod]
        [DataRow("float32", "float32", "c", 1, 6, 7001UL, false, "float32", "sha:c48b54a867db6012094ebad5faa6d8d2")]
        [DataRow("float32", "float32", "c", 2, 6, 7002UL, false, "float32", "sha:1035ead67be93fc30b4433f987ed2287")]
        [DataRow("float32", "float32", "c", 3, 6, 7003UL, false, "float32", "sha:c58ae4456249eff00b8553cc33162e99")]
        [DataRow("float32", "float32", "c", 7, 6, 7004UL, false, "float32", "sha:33e45634a62f34f103171911d67b8514")]
        [DataRow("float32", "float32", "c", 8, 6, 7005UL, false, "float32", "sha:82c88a08a4812d193bf5ce4b11d3b8bd")]
        [DataRow("float32", "float32", "c", 9, 6, 7006UL, false, "float32", "sha:2a3eb9b5d40b64a8f12fc3e1f969533f")]
        [DataRow("float32", "float32", "c", 16, 6, 7007UL, false, "float32", "sha:fe1c86a7a34cef5d68855800518c8af9")]
        [DataRow("float32", "float32", "c", 17, 6, 7008UL, false, "float32", "sha:1e0ab5b3ee90eff18c3c3ac2edf36b80")]
        [DataRow("float32", "float32", "c", 32, 6, 7009UL, false, "float32", "sha:eb09fc08c7fb4906e2f425a09afac138")]
        [DataRow("float32", "float32", "c", 33, 6, 7010UL, false, "float32", "sha:d753d3f80955194eb1d4a133f669dd01")]
        [DataRow("float32", "float32", "rev", 1, 6, 7011UL, false, "float32", "sha:9b747a05147067389fd9f4d5c05209b6")]
        [DataRow("float32", "float32", "rev", 2, 6, 7012UL, false, "float32", "sha:39961a1b1a5bc280675bfe672faa3b0d")]
        [DataRow("float32", "float32", "rev", 3, 6, 7013UL, false, "float32", "sha:d0e067688b08b3691799c84e0c2e7a1b")]
        [DataRow("float32", "float32", "rev", 7, 6, 7014UL, false, "float32", "sha:706fbb2e71962062a5f0daacd811421f")]
        [DataRow("float32", "float32", "rev", 8, 6, 7015UL, false, "float32", "sha:7c5b2252f3a76ddac58f1a2ced546e44")]
        [DataRow("float32", "float32", "rev", 9, 6, 7016UL, false, "float32", "sha:f403ce10b9dfbe616032f1209ea8e62a")]
        [DataRow("float32", "float32", "rev", 16, 6, 7017UL, false, "float32", "sha:e3a363eab2e8b494a448dd44a0468863")]
        [DataRow("float32", "float32", "rev", 17, 6, 7018UL, false, "float32", "sha:78cc3a7dfdb7961fba91be34fee6dd63")]
        [DataRow("float32", "float32", "rev", 32, 6, 7019UL, false, "float32", "sha:631a4a856ecb2d4af56a4bd352cfa696")]
        [DataRow("float32", "float32", "rev", 33, 6, 7020UL, false, "float32", "sha:be4ce145fc582ec116ee4b4ed8ff0c5f")]
        [DataRow("float32", "float32", "b2", 1, 6, 7021UL, false, "float32", "sha:76d4f0b153710752fab8bbee77fd410a")]
        [DataRow("float32", "float32", "b2", 2, 6, 7022UL, false, "float32", "sha:0518629afd0d23c7b628a70ce3cf3f37")]
        [DataRow("float32", "float32", "b2", 3, 6, 7023UL, false, "float32", "sha:1bc95ba663c086e020caa266bb3c6eca")]
        [DataRow("float32", "float32", "b2", 7, 6, 7024UL, false, "float32", "sha:5c3856240faa66cad34fd2f98eba8354")]
        [DataRow("float32", "float32", "b2", 8, 6, 7025UL, false, "float32", "sha:1f222ee46a0a8a920bb03684df256141")]
        [DataRow("float32", "float32", "b2", 9, 6, 7026UL, false, "float32", "sha:ebd0799a6597f1f10c0c023952d0f85a")]
        [DataRow("float32", "float32", "b2", 16, 6, 7027UL, false, "float32", "sha:999c697a768978e1b5a0cef38eed0132")]
        [DataRow("float32", "float32", "b2", 17, 6, 7028UL, false, "float32", "sha:11adcec57d35cfa034b8f25e0f283273")]
        [DataRow("float32", "float32", "b2", 32, 6, 7029UL, false, "float32", "sha:7fa051a75f7035a7a09376cce823c74b")]
        [DataRow("float32", "float32", "b2", 33, 6, 7030UL, false, "float32", "sha:a588563f14bca3a901191fe440fedace")]
        [DataRow("float32", "float32", "t1", 1, 6, 7031UL, false, "float32", "sha:ff2c95e2a4e0ca0120328aeba44f000e")]
        [DataRow("float32", "float32", "t1", 2, 6, 7032UL, false, "float32", "sha:9b42479a37edf7aa67798ff19ebd7368")]
        [DataRow("float32", "float32", "t1", 3, 6, 7033UL, false, "float32", "sha:2f99843c8f2fe938dd43a609d58bf4e4")]
        [DataRow("float32", "float32", "t1", 7, 6, 7034UL, false, "float32", "sha:687394a73a53f5eb845f471d69788eae")]
        [DataRow("float32", "float32", "t1", 8, 6, 7035UL, false, "float32", "sha:c78615a3cde4d92d89d504e986eb09e6")]
        [DataRow("float32", "float32", "t1", 9, 6, 7036UL, false, "float32", "sha:9b7a6eea42d0551e531de2495bb4d339")]
        [DataRow("float32", "float32", "t1", 16, 6, 7037UL, false, "float32", "sha:06a5ed680d611e79e384886b269cf98d")]
        [DataRow("float32", "float32", "t1", 17, 6, 7038UL, false, "float32", "sha:de491caf7874dadf5da1735caa71e331")]
        [DataRow("float32", "float32", "t1", 32, 6, 7039UL, false, "float32", "sha:f4cdc628d5a6c08c1e838c3f35312cc4")]
        [DataRow("float32", "float32", "t1", 33, 6, 7040UL, false, "float32", "sha:b23b7c2a2b91f5843b9979aa2ca4daa3")]
        [DataRow("float32", "float32", "tt", 1, 6, 7041UL, false, "float32", "sha:dc460628c5b934a71de04d528dd4903f")]
        [DataRow("float32", "float32", "tt", 2, 6, 7042UL, false, "float32", "sha:648ac396bae7a766e0daf16df6f6dc64")]
        [DataRow("float32", "float32", "tt", 3, 6, 7043UL, false, "float32", "sha:9efe1e38fce76c18bbdb3b1c4b88e816")]
        [DataRow("float32", "float32", "tt", 7, 6, 7044UL, false, "float32", "sha:7b53fec0c17fac5da04fb76c4616f725")]
        [DataRow("float32", "float32", "tt", 8, 6, 7045UL, false, "float32", "sha:b2352d07321710bf52c01e77ad78ecc9")]
        [DataRow("float32", "float32", "tt", 9, 6, 7046UL, false, "float32", "sha:2ce69ae8a7da72b30a3a1d8b2d571fde")]
        [DataRow("float32", "float32", "tt", 16, 6, 7047UL, false, "float32", "sha:39b817c49c8083141a889b9897ff0900")]
        [DataRow("float32", "float32", "tt", 17, 6, 7048UL, false, "float32", "sha:c14a65144b5b923b01441b75deab7c94")]
        [DataRow("float32", "float32", "tt", 32, 6, 7049UL, false, "float32", "sha:cc829e71278174171e71e8cb59414bcb")]
        [DataRow("float32", "float32", "tt", 33, 6, 7050UL, false, "float32", "sha:0ed2592b11fb46575e3434a5dfe563c9")]
        [DataRow("float32", "float32", "step", 1, 6, 7051UL, false, "float32", "sha:76baf337d096c54f84450400b5fad884")]
        [DataRow("float32", "float32", "step", 2, 6, 7052UL, false, "float32", "sha:1415e79b39368408ba83eb1d6270814b")]
        [DataRow("float32", "float32", "step", 3, 6, 7053UL, false, "float32", "sha:3e7023b2f6f2504e52472b6376f59a16")]
        [DataRow("float32", "float32", "step", 7, 6, 7054UL, false, "float32", "sha:741d9fca853a2d87de6b0ecc25a96fa4")]
        [DataRow("float32", "float32", "step", 8, 6, 7055UL, false, "float32", "sha:ce250afa66520ffee7e892a86e97d68d")]
        [DataRow("float32", "float32", "step", 9, 6, 7056UL, false, "float32", "sha:9c583c012dae115a07031ddd45392980")]
        [DataRow("float32", "float32", "step", 16, 6, 7057UL, false, "float32", "sha:2f29ecb946e374e577b50c61aa80e249")]
        [DataRow("float32", "float32", "step", 17, 6, 7058UL, false, "float32", "sha:3d841e536288dee06c1711d7a2b5c342")]
        [DataRow("float32", "float32", "step", 32, 6, 7059UL, false, "float32", "sha:6cc8fd624a0372ab69f460ad11e28dd7")]
        [DataRow("float32", "float32", "step", 33, 6, 7060UL, false, "float32", "sha:fbcf154b2f20510be861fdb97af003fa")]
        [DataRow("float32", "float32", "bcore", 1, 6, 7061UL, false, "float32", "sha:5aea0602d161d48488185a3e543d8b50")]
        [DataRow("float32", "float32", "bcore", 2, 6, 7062UL, false, "float32", "sha:8dca72fc6b0bd8f00b73288ee10fa62a")]
        [DataRow("float32", "float32", "bcore", 3, 6, 7063UL, false, "float32", "sha:5ed4a933d85677498f4f675c2c762458")]
        [DataRow("float32", "float32", "bcore", 7, 6, 7064UL, false, "float32", "sha:58f5fc98edbc8e7b56ae6affdc6523ae")]
        [DataRow("float32", "float32", "bcore", 8, 6, 7065UL, false, "float32", "sha:16beca26b866e8486f5527bd9dc7e850")]
        [DataRow("float32", "float32", "bcore", 9, 6, 7066UL, false, "float32", "sha:6db4fbd051d317615d45da7473545ada")]
        [DataRow("float32", "float32", "bcore", 16, 6, 7067UL, false, "float32", "sha:e86719a9747a90ef51a4bbf0e0813d98")]
        [DataRow("float32", "float32", "bcore", 17, 6, 7068UL, false, "float32", "sha:674f2ad08e82446ca9ed8375057d8154")]
        [DataRow("float32", "float32", "bcore", 32, 6, 7069UL, false, "float32", "sha:48d4bdfce1f971107f7d26082f1086db")]
        [DataRow("float32", "float32", "bcore", 33, 6, 7070UL, false, "float32", "sha:349a7c8d7af1cf5040291ac750c2d34d")]
        [DataRow("float32", "float32", "d3", 1, 6, 7071UL, false, "float32", "sha:88759dfe8de3d9d661e100395b5c77ff")]
        [DataRow("float32", "float32", "d3", 2, 6, 7072UL, false, "float32", "sha:01632222bc7ef502109c67c44b2e68a6")]
        [DataRow("float32", "float32", "d3", 3, 6, 7073UL, false, "float32", "sha:d06d1c981485d65c6799131a5816ad85")]
        [DataRow("float32", "float32", "d3", 7, 6, 7074UL, false, "float32", "sha:ec8b114eb550a827b55db8aa9359724c")]
        [DataRow("float32", "float32", "d3", 8, 6, 7075UL, false, "float32", "sha:3861844478871c467e250cb53debccb0")]
        [DataRow("float32", "float32", "d3", 9, 6, 7076UL, false, "float32", "sha:63f22d6f9d27d5a0848d75f3caca8b33")]
        [DataRow("float32", "float32", "d3", 16, 6, 7077UL, false, "float32", "sha:101ab8d021301191a6b24920de7ccb69")]
        [DataRow("float32", "float32", "d3", 17, 6, 7078UL, false, "float32", "sha:e3776120a6985624c6bace5a4f689a98")]
        [DataRow("float32", "float32", "d3", 32, 6, 7079UL, false, "float32", "sha:be77f97d964a9617f9b55da2b23a0caa")]
        [DataRow("float32", "float32", "d3", 33, 6, 7080UL, false, "float32", "sha:8e6cb9289b03f16296d82bc103c53581")]
        [DataRow("float32", "float32", "c", 8, 6, 7081UL, true, "float32", "hex:0000c07f0000807f01c5044532939dc20d15973df88f9847")]
        [DataRow("float32", "float32", "c", 33, 6, 7082UL, true, "float32", "hex:0000c07ffe0e04c9cd3f12475fe187c704b0404399784dca")]
        [DataRow("float32", "float32", "rev", 8, 6, 7083UL, true, "float32", "hex:0000c07f000080ff117e8dc4f6be8243274638c4ca2be6c4")]
        [DataRow("float32", "float32", "rev", 33, 6, 7084UL, true, "float32", "hex:0000c07f558808c7da93bdc8eef032c78cfe4149848e9947")]
        [DataRow("float32", "float32", "t1", 8, 6, 7085UL, true, "float32", "hex:0000c07f0000c07f334b29c5000080ff2ad05c42000080ff")]
        [DataRow("float32", "float32", "t1", 33, 6, 7086UL, true, "float32", "hex:0000c07f0000c07f128c33c7000080ffef8afa470000807f")]
        [DataRow("float32", "float32", "c", 1, 37, 7087UL, true, "float32", "hex:e987e4c30000c07f000000006e47ae3fba7a93bc0000807f25a484bb6a25d6bc0a710ac00000807f3a531ebe19f9a341ae8d19bd2c4d3b4570900a3d4d6787c7489cdebb6be551c6d8d14645f3812f3cb40aa0bb9de693435165a539a3ed233e190508bd79668c3cdbf2aa41508372bd3de81ac714c39eba5a5e54b92a6e2240f26e103d6fc80442bf22a8c1d7f019bdc4eb88bf")]
        [DataRow("float32", "float32", "b2", 1, 37, 7088UL, true, "float32", "hex:daa182440000c07f00000000442f7bbd10c4dd3e0000807fe94cdc3d19d631c3ca94dbc0000080fff14cc0402d1915be2023a33c2b21e13f53ef42bce70105c4cbe89a3d6305c8c1b2dfafbba30b52430d2933bc9e0484bdcbc5dfbd96ada7bdc73a80bd922607c308b1bcbee75f473c96008fbd0037dfbe6514073c5f01c0bdf4487dc267770543c1e68dbf4b1abe41cbf071c0")]
        [DataRow("float64", "float64", "c", 1, 6, 7089UL, false, "float64", "sha:a56f3d9ffc659dcb315f0d6fb09e3f71")]
        [DataRow("float64", "float64", "c", 2, 6, 7090UL, false, "float64", "sha:bf3f280b5f8090e6cfc84fd810a553cc")]
        [DataRow("float64", "float64", "c", 3, 6, 7091UL, false, "float64", "sha:cf21bdf56e4ed13c3b1a4a7bc77535b7")]
        [DataRow("float64", "float64", "c", 7, 6, 7092UL, false, "float64", "sha:d19cc28a13d2315a11f66a1e1feff76b")]
        [DataRow("float64", "float64", "c", 8, 6, 7093UL, false, "float64", "sha:268e1257ba6704fe18f350955dfd1207")]
        [DataRow("float64", "float64", "c", 9, 6, 7094UL, false, "float64", "sha:3589fcb627fdd8ac4370aa5a61b12b91")]
        [DataRow("float64", "float64", "c", 16, 6, 7095UL, false, "float64", "sha:c64b2359ce36a3ddfb5380fc8fba36ee")]
        [DataRow("float64", "float64", "c", 17, 6, 7096UL, false, "float64", "sha:1c60e940ed10faa62a3b40d0abaaa9da")]
        [DataRow("float64", "float64", "c", 32, 6, 7097UL, false, "float64", "sha:943cf7d07458dc2903d1601114b720a4")]
        [DataRow("float64", "float64", "c", 33, 6, 7098UL, false, "float64", "sha:e1dc41c547c44e773fc4a218fad3dad6")]
        [DataRow("float64", "float64", "rev", 1, 6, 7099UL, false, "float64", "sha:9c42f676409598664e7724e3753c14c3")]
        [DataRow("float64", "float64", "rev", 2, 6, 7100UL, false, "float64", "sha:cc6a4a9ce1e0795cc8dbb8ca0ac6e73e")]
        [DataRow("float64", "float64", "rev", 3, 6, 7101UL, false, "float64", "sha:5fa47f5df68d4f00e91b46ec8752e11c")]
        [DataRow("float64", "float64", "rev", 7, 6, 7102UL, false, "float64", "sha:11378ce24a89f2f957151d43efc5877d")]
        [DataRow("float64", "float64", "rev", 8, 6, 7103UL, false, "float64", "sha:81b26d7dd0e79bf07f233ee8b8e72553")]
        [DataRow("float64", "float64", "rev", 9, 6, 7104UL, false, "float64", "sha:77f54f8b331acde90c856f7291f84f7d")]
        [DataRow("float64", "float64", "rev", 16, 6, 7105UL, false, "float64", "sha:86f4a5b6557e33a24cd8c88185b01de6")]
        [DataRow("float64", "float64", "rev", 17, 6, 7106UL, false, "float64", "sha:d6070197d9e4a32be9619dce9809d9ca")]
        [DataRow("float64", "float64", "rev", 32, 6, 7107UL, false, "float64", "sha:745bb39479b82b5ad688e545980a6930")]
        [DataRow("float64", "float64", "rev", 33, 6, 7108UL, false, "float64", "sha:3e88cdde47c973745dfa79b7b3d13345")]
        [DataRow("float64", "float64", "b2", 1, 6, 7109UL, false, "float64", "sha:abd8503f7e3c037d50999b72ed58397f")]
        [DataRow("float64", "float64", "b2", 2, 6, 7110UL, false, "float64", "sha:f27d4bd75ee65bfec2bea7a375238265")]
        [DataRow("float64", "float64", "b2", 3, 6, 7111UL, false, "float64", "sha:04d9c414df52f731c6bea1131856553d")]
        [DataRow("float64", "float64", "b2", 7, 6, 7112UL, false, "float64", "sha:188d1257d9f25db9125defd76a07fd5a")]
        [DataRow("float64", "float64", "b2", 8, 6, 7113UL, false, "float64", "sha:336a047580eed0802c629993688e2a35")]
        [DataRow("float64", "float64", "b2", 9, 6, 7114UL, false, "float64", "sha:1d9491b6335c4670f9da72dc40d3fefe")]
        [DataRow("float64", "float64", "b2", 16, 6, 7115UL, false, "float64", "sha:40db85531dc219112a8286de0912b22d")]
        [DataRow("float64", "float64", "b2", 17, 6, 7116UL, false, "float64", "sha:089f54747bd36d1a5b0bfa3f834d2599")]
        [DataRow("float64", "float64", "b2", 32, 6, 7117UL, false, "float64", "sha:d681edab5c873a650cd6da20ae3d040d")]
        [DataRow("float64", "float64", "b2", 33, 6, 7118UL, false, "float64", "sha:ce8904a1ca6e79353ec54551b8c89b3f")]
        [DataRow("float64", "float64", "t1", 1, 6, 7119UL, false, "float64", "sha:4bba6b1ab604041fb61ec445b35787e1")]
        [DataRow("float64", "float64", "t1", 2, 6, 7120UL, false, "float64", "sha:40068a4b5369d20afe344e7f3938a295")]
        [DataRow("float64", "float64", "t1", 3, 6, 7121UL, false, "float64", "sha:02d577ad0b871b7350c8044731c5e5d0")]
        [DataRow("float64", "float64", "t1", 7, 6, 7122UL, false, "float64", "sha:25448830828790fbf5804b8a1b7ebc93")]
        [DataRow("float64", "float64", "t1", 8, 6, 7123UL, false, "float64", "sha:ec3bdfa09bec9675ec5ac7e16b88c5b3")]
        [DataRow("float64", "float64", "t1", 9, 6, 7124UL, false, "float64", "sha:6a5c42bad8938141b0129825dd9c4143")]
        [DataRow("float64", "float64", "t1", 16, 6, 7125UL, false, "float64", "sha:74b80aa30eab16a4d77a07444d4635a3")]
        [DataRow("float64", "float64", "t1", 17, 6, 7126UL, false, "float64", "sha:a1e57adfe56a33f6021e57dc10ff9f28")]
        [DataRow("float64", "float64", "t1", 32, 6, 7127UL, false, "float64", "sha:a6481c6b783307cccba9b9ac5c538d6f")]
        [DataRow("float64", "float64", "t1", 33, 6, 7128UL, false, "float64", "sha:be65883bbbf1f271ff38ff34aeddd6f6")]
        [DataRow("float64", "float64", "tt", 1, 6, 7129UL, false, "float64", "sha:7edbb09eb3bdaf99a61598cc335f829a")]
        [DataRow("float64", "float64", "tt", 2, 6, 7130UL, false, "float64", "sha:bedb9074a49b07fedee3065e4aea8f13")]
        [DataRow("float64", "float64", "tt", 3, 6, 7131UL, false, "float64", "sha:6b0498561c6eb967df3cc62a7678b863")]
        [DataRow("float64", "float64", "tt", 7, 6, 7132UL, false, "float64", "sha:48af0dbaf8c439dcc717c5cbf2748839")]
        [DataRow("float64", "float64", "tt", 8, 6, 7133UL, false, "float64", "sha:53d9839d3d9ef77a3957e69aa27cef0e")]
        [DataRow("float64", "float64", "tt", 9, 6, 7134UL, false, "float64", "sha:5220700b846c6aca8ecaddac004048ac")]
        [DataRow("float64", "float64", "tt", 16, 6, 7135UL, false, "float64", "sha:851147daa234879d86b5b27c16ae8e2e")]
        [DataRow("float64", "float64", "tt", 17, 6, 7136UL, false, "float64", "sha:c50656c8d37f538996314788869f3d86")]
        [DataRow("float64", "float64", "tt", 32, 6, 7137UL, false, "float64", "sha:ab7194aec5b086fefb2a1e0d91899300")]
        [DataRow("float64", "float64", "tt", 33, 6, 7138UL, false, "float64", "sha:36ffd4b8805ca610cfabee14619c91d2")]
        [DataRow("float64", "float64", "step", 1, 6, 7139UL, false, "float64", "sha:0526a28bc73ddb3fc9d899fec5b8f493")]
        [DataRow("float64", "float64", "step", 2, 6, 7140UL, false, "float64", "sha:d7a4994bb399ac46a967bb4a722837f5")]
        [DataRow("float64", "float64", "step", 3, 6, 7141UL, false, "float64", "sha:e8217addec3a8f01b2670870e3f0e33a")]
        [DataRow("float64", "float64", "step", 7, 6, 7142UL, false, "float64", "sha:35ca1548d9776148dcc117b311e876b9")]
        [DataRow("float64", "float64", "step", 8, 6, 7143UL, false, "float64", "sha:6340399ea4fcc5013bb1b6a14c2e783c")]
        [DataRow("float64", "float64", "step", 9, 6, 7144UL, false, "float64", "sha:c4d79c35552aa7aa9578996811039a9b")]
        [DataRow("float64", "float64", "step", 16, 6, 7145UL, false, "float64", "sha:78009585d5fa00489afa12af205b7df1")]
        [DataRow("float64", "float64", "step", 17, 6, 7146UL, false, "float64", "sha:b0591a63241f28345491a20b079e20ab")]
        [DataRow("float64", "float64", "step", 32, 6, 7147UL, false, "float64", "sha:9b33f2506cd533c6ed2818c0fae0f3f7")]
        [DataRow("float64", "float64", "step", 33, 6, 7148UL, false, "float64", "sha:48f13ea639c33099f7af516d0e632507")]
        [DataRow("float64", "float64", "bcore", 1, 6, 7149UL, false, "float64", "sha:dd2fdd3b2d2e1ab4a19ee829871f9fc9")]
        [DataRow("float64", "float64", "bcore", 2, 6, 7150UL, false, "float64", "sha:1df753e5d40922ff114e918087906de7")]
        [DataRow("float64", "float64", "bcore", 3, 6, 7151UL, false, "float64", "sha:57c179bb6b82ded85b3fb614fde69c7b")]
        [DataRow("float64", "float64", "bcore", 7, 6, 7152UL, false, "float64", "sha:33f088ff8729105bff0635f27487ccf4")]
        [DataRow("float64", "float64", "bcore", 8, 6, 7153UL, false, "float64", "sha:9d91d2bfad8c02cb2396948a41bc266f")]
        [DataRow("float64", "float64", "bcore", 9, 6, 7154UL, false, "float64", "sha:134c227909cc1914ff455140e63305f6")]
        [DataRow("float64", "float64", "bcore", 16, 6, 7155UL, false, "float64", "sha:28942968ed92da541d7e6647a900b067")]
        [DataRow("float64", "float64", "bcore", 17, 6, 7156UL, false, "float64", "sha:733e84fe2f3728592123157cb1d27f62")]
        [DataRow("float64", "float64", "bcore", 32, 6, 7157UL, false, "float64", "sha:332e92b473e4cb76a1dfd061284e4073")]
        [DataRow("float64", "float64", "bcore", 33, 6, 7158UL, false, "float64", "sha:3d9c6f911b523d9f0acbb7600257570e")]
        [DataRow("float64", "float64", "d3", 1, 6, 7159UL, false, "float64", "sha:eac7226cd0a33883980e183bff67e9d3")]
        [DataRow("float64", "float64", "d3", 2, 6, 7160UL, false, "float64", "sha:9d70ef44eacb6f4dadac805e529182ff")]
        [DataRow("float64", "float64", "d3", 3, 6, 7161UL, false, "float64", "sha:08c55fdf4d7478a46bc7a46966e7def4")]
        [DataRow("float64", "float64", "d3", 7, 6, 7162UL, false, "float64", "sha:61f39fa8bd67786bb11472c2e58f1a52")]
        [DataRow("float64", "float64", "d3", 8, 6, 7163UL, false, "float64", "sha:480679512087ed87557ad704f77cb7b0")]
        [DataRow("float64", "float64", "d3", 9, 6, 7164UL, false, "float64", "sha:4e07522820959089ef4546dd55eb2673")]
        [DataRow("float64", "float64", "d3", 16, 6, 7165UL, false, "float64", "sha:016ee45f73db5159b1d1de49f78065e4")]
        [DataRow("float64", "float64", "d3", 17, 6, 7166UL, false, "float64", "sha:a53c3e58f2bff08a09e4fb4e84fc424b")]
        [DataRow("float64", "float64", "d3", 32, 6, 7167UL, false, "float64", "sha:c01c175005d665363af06488f1eefab9")]
        [DataRow("float64", "float64", "d3", 33, 6, 7168UL, false, "float64", "sha:fdc264524011c624433b5cbf116e32da")]
        [DataRow("float64", "float64", "c", 8, 6, 7169UL, true, "float64", "hex:000000000000f87f000000000000f07fc14f2241ce2c8940e090d4363edeccc0211bc058f9ba3041fe1b6b77eb44a7c0")]
        [DataRow("float64", "float64", "c", 33, 6, 7170UL, true, "float64", "hex:000000000000f87f314988485d8a3141d0f0eef4a8d4ddc06e78055c22e0f0404621e657fdc4214150e1c45533ad3241")]
        [DataRow("float64", "float64", "rev", 8, 6, 7171UL, true, "float64", "hex:000000000000f87f000000000000f8ff0d45d2a64fe270c07a401107131897c0f3efb645999650c0e0263ca34ffe6cc0")]
        [DataRow("float64", "float64", "rev", 33, 6, 7172UL, true, "float64", "hex:000000000000f87f0345fc0ef88d7740c51d99efb341d040f108487b94d10b41ad0eba11726db34057696086cb8825c1")]
        [DataRow("float64", "float64", "t1", 8, 6, 7173UL, true, "float64", "hex:000000000000f87f000000000000f87f93fa97fa698db940000000000000f0ffa55c742e3e945240000000000000f07f")]
        [DataRow("float64", "float64", "t1", 33, 6, 7174UL, true, "float64", "hex:000000000000f87f000000000000f87fbc022db9a311ebc0000000000000f0ff333c8d359e5867c0000000000000f0ff")]
        [DataRow("float64", "float64", "c", 1, 37, 7175UL, true, "float64", "hex:600d95cb9dd2bbc0000000000000f87f000000000000000008f7580c01388740cfdd01c0b1864b40000000000000f07f1d4f225925d76ac0a55002f0ccc93440cf2d2c24e0ebe63e000000000000f07f9ab07b5a452da2bfc668b130c1831840f3eb51e2eb5f54bfca073b1c41dd5140228a0d97ed94b1bff57e67370950714052d155445fab97bf2e010613fad0d9be8804e6697fa8ea40123326d75aadeebff567240a09f37ebfe42c8375f3435fbfdced59938484b13f5f9ccd2da776f73f8b8640c530bc64bf32fb29e9d80593bf80f362d2a74a8fbfd634e51dab9360bf9a81a907dc3b854056c46818ca02ecbf8477a929cefc23c15d3abc519e840741dfd97bf1d9d3bd3f2194b1a37a04dc3f2af850de49b025c122a6ee2252e1d83f65a9f8eff148e83e")]
        [DataRow("float64", "float64", "b2", 1, 37, 7176UL, true, "float64", "hex:3422f5887649d040000000000000f87f0000000000000000b8c46871b937e7bf42d551f85f373040000000000000f0ff76eed84013ac3140145410a5fb27b13f231c7548ca9735c0000000000000f07f40e1ea1401a49ec0e0514643a48a04c09b9d60992f5b2e40b6ec6b22a070e63fb106035c8916ed3f697b370198484dc03240730878d5fbbfd313262ae3c0d83fff3f3b8ace8d2ac04c653a2967a697bf878f6c61c797db3f5ab1df0ad5330bc0ec13b905083ea43fa450f916b66b8b403676db1eb762af4006f9aa84adf0c340f382cde11347ed3fcfca784fc7bbda3f88939d3293172840874fce12c6f79e3f969b5d7c1af66cc0e538e915389510c0505956f9e728d8bf76e6e9667c76a5c0d3a753b9ac9a2c40177d11d3cd625040fe849f93cd2ba640")]
        [DataRow("complex128", "complex128", "c", 1, 6, 7177UL, false, "complex128", "sha:079f607a923e8df38bfefd34578bb10d")]
        [DataRow("complex128", "complex128", "c", 2, 6, 7178UL, false, "complex128", "sha:d6ce8b81039532e83fee57f2d69ab082")]
        [DataRow("complex128", "complex128", "c", 3, 6, 7179UL, false, "complex128", "sha:eb1c853f5556faf9fcdefc7ffb3633b3")]
        [DataRow("complex128", "complex128", "c", 7, 6, 7180UL, false, "complex128", "sha:b1d5555847cbfa55ff549fe308a06ba4")]
        [DataRow("complex128", "complex128", "c", 8, 6, 7181UL, false, "complex128", "sha:346690f2d632cd8db0673ec9894d04e0")]
        [DataRow("complex128", "complex128", "c", 9, 6, 7182UL, false, "complex128", "sha:1ac0649ac1b319ded6a849d3afd348f8")]
        [DataRow("complex128", "complex128", "c", 16, 6, 7183UL, false, "complex128", "sha:7cf4d4a2d46f9367b60fb6612d449883")]
        [DataRow("complex128", "complex128", "c", 17, 6, 7184UL, false, "complex128", "sha:467602734b667f57ccb51cb749e44ef0")]
        [DataRow("complex128", "complex128", "c", 32, 6, 7185UL, false, "complex128", "sha:6f0f7ee8dc307d91536543e6394ebe85")]
        [DataRow("complex128", "complex128", "c", 33, 6, 7186UL, false, "complex128", "sha:97209de294ae5eca91e1a2d6b6d90f9a")]
        [DataRow("complex128", "complex128", "rev", 1, 6, 7187UL, false, "complex128", "sha:a6adaa341d031f190cd735ac12bc9380")]
        [DataRow("complex128", "complex128", "rev", 2, 6, 7188UL, false, "complex128", "sha:162cf2686ec496204949211410f7776e")]
        [DataRow("complex128", "complex128", "rev", 3, 6, 7189UL, false, "complex128", "sha:11e783b0d4cb81c2667f8462d467157f")]
        [DataRow("complex128", "complex128", "rev", 7, 6, 7190UL, false, "complex128", "sha:19992f1cf4d5844f9c3248653a4c9bae")]
        [DataRow("complex128", "complex128", "rev", 8, 6, 7191UL, false, "complex128", "sha:2f14383586f8267c83bc23264d384cf6")]
        [DataRow("complex128", "complex128", "rev", 9, 6, 7192UL, false, "complex128", "sha:9f2c882f5069e896f93a1145b453d7c2")]
        [DataRow("complex128", "complex128", "rev", 16, 6, 7193UL, false, "complex128", "sha:c19fde57c73a468b44ff62b9f3ce9936")]
        [DataRow("complex128", "complex128", "rev", 17, 6, 7194UL, false, "complex128", "sha:362c5a5110506ffe593c0bd239648d9c")]
        [DataRow("complex128", "complex128", "rev", 32, 6, 7195UL, false, "complex128", "sha:3abd08080ae12cadf1e43e3a1fc0cc46")]
        [DataRow("complex128", "complex128", "rev", 33, 6, 7196UL, false, "complex128", "sha:2e4b933c36d521390c4866bcd2f7da76")]
        [DataRow("complex128", "complex128", "b2", 1, 6, 7197UL, false, "complex128", "sha:e9799e13df730c0ea73f77558d352243")]
        [DataRow("complex128", "complex128", "b2", 2, 6, 7198UL, false, "complex128", "sha:44791ea691f6a93519e8456fca4c8f65")]
        [DataRow("complex128", "complex128", "b2", 3, 6, 7199UL, false, "complex128", "sha:76a96ec3f935d256731e0dcfabfed235")]
        [DataRow("complex128", "complex128", "b2", 7, 6, 7200UL, false, "complex128", "sha:71e10021bb58c8da8dcd207340425932")]
        [DataRow("complex128", "complex128", "b2", 8, 6, 7201UL, false, "complex128", "sha:e5f356ec39ed998cb8246313e113f396")]
        [DataRow("complex128", "complex128", "b2", 9, 6, 7202UL, false, "complex128", "sha:dc77a947b4e45748e89eb76e871058e1")]
        [DataRow("complex128", "complex128", "b2", 16, 6, 7203UL, false, "complex128", "sha:33f490d2663f5a433597b6b9263ecd9f")]
        [DataRow("complex128", "complex128", "b2", 17, 6, 7204UL, false, "complex128", "sha:2d0261da924362e8e0a7907a00943e08")]
        [DataRow("complex128", "complex128", "b2", 32, 6, 7205UL, false, "complex128", "sha:924325940ce7d912ec9b89b2d8ef5d29")]
        [DataRow("complex128", "complex128", "b2", 33, 6, 7206UL, false, "complex128", "sha:9227a8dfffe2d0ecc08c524650ada80b")]
        [DataRow("complex128", "complex128", "t1", 1, 6, 7207UL, false, "complex128", "sha:5402224491a4115f1033de4335909400")]
        [DataRow("complex128", "complex128", "t1", 2, 6, 7208UL, false, "complex128", "sha:a70a97031dcc25fe9030f33a2e993013")]
        [DataRow("complex128", "complex128", "t1", 3, 6, 7209UL, false, "complex128", "sha:b9a63775783a4063c44595c0c399e9b7")]
        [DataRow("complex128", "complex128", "t1", 7, 6, 7210UL, false, "complex128", "sha:43bfa0981c76320c18e63f3a967d6f38")]
        [DataRow("complex128", "complex128", "t1", 8, 6, 7211UL, false, "complex128", "sha:5b2320aa5918cbb23ed9ef76ee677127")]
        [DataRow("complex128", "complex128", "t1", 9, 6, 7212UL, false, "complex128", "sha:e93df6c26e6c5e4817aed2ee654eec5b")]
        [DataRow("complex128", "complex128", "t1", 16, 6, 7213UL, false, "complex128", "sha:3c9e97197fee90c7663643684b2a1ba0")]
        [DataRow("complex128", "complex128", "t1", 17, 6, 7214UL, false, "complex128", "sha:d3f75fbfef7c05f4f1867715560f90ce")]
        [DataRow("complex128", "complex128", "t1", 32, 6, 7215UL, false, "complex128", "sha:10420d048d23321fe7ece9f8cbf9cf8d")]
        [DataRow("complex128", "complex128", "t1", 33, 6, 7216UL, false, "complex128", "sha:e57da8ca818ca4ac14cdf476171b1757")]
        [DataRow("complex128", "complex128", "tt", 1, 6, 7217UL, false, "complex128", "sha:24ebdeff2963fc7ec71c45df48a9fe68")]
        [DataRow("complex128", "complex128", "tt", 2, 6, 7218UL, false, "complex128", "sha:97865200b167233c1ed0ef32792a90ce")]
        [DataRow("complex128", "complex128", "tt", 3, 6, 7219UL, false, "complex128", "sha:d1e94c0988a36e7c7d6d1bcf23c8a440")]
        [DataRow("complex128", "complex128", "tt", 7, 6, 7220UL, false, "complex128", "sha:36dc499627c0aa4b6291bbce837d2198")]
        [DataRow("complex128", "complex128", "tt", 8, 6, 7221UL, false, "complex128", "sha:33bf1642bcb9ad496ae1ad604e993283")]
        [DataRow("complex128", "complex128", "tt", 9, 6, 7222UL, false, "complex128", "sha:863b4372c5dfe08472c0cd8db199d0ab")]
        [DataRow("complex128", "complex128", "tt", 16, 6, 7223UL, false, "complex128", "sha:0ab96555d63671ba112824d1a991d2a0")]
        [DataRow("complex128", "complex128", "tt", 17, 6, 7224UL, false, "complex128", "sha:f536a9bd25408ce1af0c613fbd013d96")]
        [DataRow("complex128", "complex128", "tt", 32, 6, 7225UL, false, "complex128", "sha:6944964cfdd1ed5aef94eab7de4bfd73")]
        [DataRow("complex128", "complex128", "tt", 33, 6, 7226UL, false, "complex128", "sha:3461932b8b96bff5009487d62cbbf323")]
        [DataRow("complex128", "complex128", "step", 1, 6, 7227UL, false, "complex128", "sha:b89178a72d53e0371f7e36b626ac6ff9")]
        [DataRow("complex128", "complex128", "step", 2, 6, 7228UL, false, "complex128", "sha:41330c7aafe5c7b1bff16c22853747f1")]
        [DataRow("complex128", "complex128", "step", 3, 6, 7229UL, false, "complex128", "sha:241b1349378d98ee74db0029c28c8e9f")]
        [DataRow("complex128", "complex128", "step", 7, 6, 7230UL, false, "complex128", "sha:56d0cc72f67101d81bb74946fed2f48a")]
        [DataRow("complex128", "complex128", "step", 8, 6, 7231UL, false, "complex128", "sha:ae243ef2c819a921aaeca027eb3d0e2e")]
        [DataRow("complex128", "complex128", "step", 9, 6, 7232UL, false, "complex128", "sha:e46c91eca99488f248053b53f828a5d2")]
        [DataRow("complex128", "complex128", "step", 16, 6, 7233UL, false, "complex128", "sha:898bd6f083d59780376c23e5a21ca7c7")]
        [DataRow("complex128", "complex128", "step", 17, 6, 7234UL, false, "complex128", "sha:d4cf801f9291ff54e087ce082b63b5c9")]
        [DataRow("complex128", "complex128", "step", 32, 6, 7235UL, false, "complex128", "sha:ab628661248fb9110764eba3b49fa552")]
        [DataRow("complex128", "complex128", "step", 33, 6, 7236UL, false, "complex128", "sha:8a72698eb527ada47b8af8ab7185a915")]
        [DataRow("complex128", "complex128", "bcore", 1, 6, 7237UL, false, "complex128", "sha:f9b4ab7e2155b9f9eefcabdbf8a19770")]
        [DataRow("complex128", "complex128", "bcore", 2, 6, 7238UL, false, "complex128", "sha:5baf9fb5771d174ab2f6ec2b3cc683fe")]
        [DataRow("complex128", "complex128", "bcore", 3, 6, 7239UL, false, "complex128", "sha:b4be4e8cf5ac391e1965fddc70834621")]
        [DataRow("complex128", "complex128", "bcore", 7, 6, 7240UL, false, "complex128", "sha:62e8ef73dab43247246095688b55bb6f")]
        [DataRow("complex128", "complex128", "bcore", 8, 6, 7241UL, false, "complex128", "sha:ca72527e28d5b08192abfa916975af8e")]
        [DataRow("complex128", "complex128", "bcore", 9, 6, 7242UL, false, "complex128", "sha:518ab9fc04196b0c3d5f6a4deb7df63e")]
        [DataRow("complex128", "complex128", "bcore", 16, 6, 7243UL, false, "complex128", "sha:51b4d19e372b020ae6c98f1d177e3211")]
        [DataRow("complex128", "complex128", "bcore", 17, 6, 7244UL, false, "complex128", "sha:d5817871a6b881b50322020e12cdb855")]
        [DataRow("complex128", "complex128", "bcore", 32, 6, 7245UL, false, "complex128", "sha:982399aef5b499202bd32d00ad94e96e")]
        [DataRow("complex128", "complex128", "bcore", 33, 6, 7246UL, false, "complex128", "sha:34fd90ced893cd3193f4ab27a76d956f")]
        [DataRow("complex128", "complex128", "d3", 1, 6, 7247UL, false, "complex128", "sha:ce7d00a7cc77d19c7f9ba9d289e15b42")]
        [DataRow("complex128", "complex128", "d3", 2, 6, 7248UL, false, "complex128", "sha:4e64a08cb793dded7ae5ceb0c1c761b4")]
        [DataRow("complex128", "complex128", "d3", 3, 6, 7249UL, false, "complex128", "sha:4a9c59fc4a16186a19f36c0a2b61340b")]
        [DataRow("complex128", "complex128", "d3", 7, 6, 7250UL, false, "complex128", "sha:2d0d6b753dbf570971c72653852dada9")]
        [DataRow("complex128", "complex128", "d3", 8, 6, 7251UL, false, "complex128", "sha:deb504f7de612d46c5f214869c02f486")]
        [DataRow("complex128", "complex128", "d3", 9, 6, 7252UL, false, "complex128", "sha:cfaf0344f03d39a3df6fbfb317be378b")]
        [DataRow("complex128", "complex128", "d3", 16, 6, 7253UL, false, "complex128", "sha:45c88205fd8ad08ad1318a1ec2ca35f8")]
        [DataRow("complex128", "complex128", "d3", 17, 6, 7254UL, false, "complex128", "sha:56534b828720acf768f4d45f703147e6")]
        [DataRow("complex128", "complex128", "d3", 32, 6, 7255UL, false, "complex128", "sha:6b70883f07d2b7f245688f91a725ba25")]
        [DataRow("complex128", "complex128", "d3", 33, 6, 7256UL, false, "complex128", "sha:37b7dd1dd0eaea5a9fd8b95078d4d1be")]
        [DataRow("complex128", "complex128", "c", 8, 6, 7257UL, true, "complex128", "hex:000000000000f87f000000000000f8ff000000000000f8ff000000000000f8ff93b964637551a24000792d756900ac401bb4234e02679640f77a93d35955b740705c17e366fba0c0fac8d3e472dfd4c04f897bf1677370c0dc4084625da7d840")]
        [DataRow("complex128", "complex128", "c", 33, 6, 7258UL, true, "complex128", "hex:000000000000f87f000000000000f8ff8dd9ca0ddb97d3c07e0daf528d8bd34068d630bb5c9ed8406e229b88fe6a264153b69069b9450dc1c61fefc92dbf09c1b4e30225935ed44038dc136cb164f0c03e1cfedd9af02a41d86856c9a9500141")]
        [DataRow("complex128", "complex128", "rev", 8, 6, 7259UL, true, "complex128", "hex:000000000000f87f000000000000f87f000000000000f0ff000000000000f0fff648a6bdf2877640683b88e1b90ea5c0616e9b6ffd3d17400d56fb90ca2085c0948b65e883a3354062c8948292a58b40d167d51a8601bc400730e73d75014040")]
        [DataRow("complex128", "complex128", "rev", 33, 6, 7260UL, true, "complex128", "hex:000000000000f87f000000000000f87f42e71c3da3781941e901ab5c55112641918a7d1905361a41341249833587dac02d6c1b5c5cea30413c0218af6551c440d32a20aae4693ec11840756ec324f04099e837e3641e41c15b764471c14340c1")]
        [DataRow("complex128", "complex128", "t1", 8, 6, 7261UL, true, "complex128", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87f74b1d9f93fc6dac07c115214afe7c0c0000000000000f8ff000000000000f0ffeb83d7a1044c0dc0f2f3feccf8042bc0000000000000f8ff000000000000f0ff")]
        [DataRow("complex128", "complex128", "t1", 33, 6, 7262UL, true, "complex128", "hex:000000000000f87f000000000000f87f000000000000f87f000000000000f87fd83f3beba3d200c1d01b8aaf57b028c1000000000000f8ff000000000000f0ff104ed9fb297ee1401e089c8d723f25c1000000000000f8ff000000000000f07f")]
        [DataRow("complex128", "complex128", "c", 1, 37, 7263UL, true, "complex128", "hex:75a1f7a72c465640ecfb49e4222f5ac0000000000000f87f000000000000f87f00000000000000000000000000000000253376762dd327c00ec8a32ac25ff53fcec2774ca83289bf98146fa9382cd4bf000000000000f07f000000000000f8ff3380e889c7b4c8c0ac255389e26455c01b25f708fd35aac0ee385934d04d51403ddf70a833853340aff97e86d9fe00c0000000000000f07f000000000000f8ff4059203b51b024c05e4c76a013d53bc08f117990ba48e1bfa1703f94fa677e40bbd394e88923fabf4a2bc110f483d8bfee5f8ec2d149a140f8fa90d80fdd3141489f94256232e13f94164525c80cc0bf82203843ca488040d271a262f03100406eb1f965ff5c1ebf527a6dba0c8e243f8b5e37a8425a3c405cc40c13c13ce73f41b32d20fb3b24405acf73de3adcd13f47ea093f7322db3f189eb9d0bcf3e8bf6f7a9d1d1e15efc0b59d72bc896a3bc19c5e40b91f5660bf1c21748b3a58843fe8020af7af918c40a1ab3a8d86118240bcfae5a0046dd13f43b214fade71b23f61a38092c271e1bf7a6b62b1ea983040cc151756fd8492c0ab8a388c5a777340f6c1701a155a274000dafbe076535740cef18e8a6668ec3f0df3af3469b744c05cef09b7800d5040f22d958268ddf24072dd9abd6bd0b23f688022ec02dfd13f300aca1f7a4d21c020ec3864b35e54c0fa39a75aed21f4bffe0dd89e003333401e9a774ce1e34e40a40bc332890773c09ef0e43965e82040b48d700a4d5cd440be1c041c34c058402914402808c005406e9ca0c3855991bfb1f58dd65123c0bff26e71fd78e94040de2aa9d3fcc920c0")]
        [DataRow("complex128", "complex128", "b2", 1, 37, 7264UL, true, "complex128", "hex:4280ee710b7181bf8bfa09842415763f000000000000f87f000000000000f87f00000000000000000000000000000000246809c61f6cb8bf2803f23170f8febf8b39f34fdc3c22c0e0e0054c17b5b1bf000000000000f0ff000000000000f0ffabb22271932bd8bfc71ab8a82a4fb23f6e1c1a5c287422c0fbdcd1d2af11e93f863866e3a016f43f6dece7d5f4cdd9bf000000000000f07f000000000000f07f5c72f78f70efa1c06e96c4bcfba026c06f4177a0dc59004093a4879a3befe23f7076f146b76410408c1dfff12b9471bf442a636fa4be6340ed39127b04ce8ec082b96e9ba94ae8bf3eaa272cc671624030588ddc9b38a9bf2b56cc873a285ebf6cda12a37b9cc23f9c76421d693e51bfae99bdfdb5639abfd898e3ae32921940a09aeae749147ebf9b4028104d4e28c01055d989da8011c0d1c8220d7da797401deee0d53b95ef3f89e4a26b66234140460eb8b45a3972bf205513ae37296e3fbe9eea0304d91fc032d8f9a0f2b3c3bfb164de92c3b186bf8ad2b176913ba03fd00add65c8c3c5bf1a3fdd2ad634344088ab153a93ac80c08cea95694423fdbf9131a5d4b65069c0f6cc60c90ab561c049ea0693be5293bf6c6f7bc7b71b7fbfe21fe9111dc1f13fe064af8ed4386d3f222b6072b90f90400efd5ef3794214406780730f4b4c71bfb72717e1c388a13f019553b1c5538ec06673aded9a2713c0f0d6a4de919600c074ff3a7f039b82402d98936ca93730c071816634c30b4a408a4fcf4782f9d6bf3ed3833c42ad91bf293bbdae026a534002f0b4196e9540c0f86c4db9006c4040485227d69cf9c43f")]
        [DataRow("float32", "float64", "c", 1, 6, 7265UL, false, "float64", "sha:a7f9c2a021f09381114d517c130c7a62")]
        [DataRow("float32", "float64", "rev", 1, 6, 7266UL, false, "float64", "sha:f3249d1de08c59b6ac29c5faa74c1fc1")]
        [DataRow("float32", "float64", "b2", 1, 6, 7267UL, false, "float64", "sha:189c2cb6489ef95fc73845d32f8eab94")]
        [DataRow("float32", "float64", "t1", 1, 6, 7268UL, false, "float64", "sha:d3961614d7f4ea956e62035bbacb95c5")]
        [DataRow("float32", "float64", "tt", 1, 6, 7269UL, false, "float64", "sha:82c2b6a6205c0b25682c0ad5c9be3891")]
        [DataRow("float32", "float64", "c", 5, 6, 7270UL, false, "float64", "sha:3e38eac4c6d3e85591aa4638aa24a106")]
        [DataRow("float32", "float64", "rev", 5, 6, 7271UL, false, "float64", "sha:de2e065e151f11307c633aaff11b1ddd")]
        [DataRow("float32", "float64", "b2", 5, 6, 7272UL, false, "float64", "sha:2e1838bba0a902ebd64b5225d52069ec")]
        [DataRow("float32", "float64", "t1", 5, 6, 7273UL, false, "float64", "sha:b5bd20bcfb16fb16d879387078c1cb1e")]
        [DataRow("float32", "float64", "tt", 5, 6, 7274UL, false, "float64", "sha:f6d9dd039d653d7926a495082b14c8e8")]
        [DataRow("float32", "float64", "c", 40, 6, 7275UL, false, "float64", "sha:3719d5e88a9ecb407bd185252b9b8664")]
        [DataRow("float32", "float64", "rev", 40, 6, 7276UL, false, "float64", "sha:433eedcf219b2aab022a06fa2becae69")]
        [DataRow("float32", "float64", "b2", 40, 6, 7277UL, false, "float64", "sha:87213ac060dea9eda881022a6261e160")]
        [DataRow("float32", "float64", "t1", 40, 6, 7278UL, false, "float64", "sha:f8fca1490af44ed54c8205d30e88c719")]
        [DataRow("float32", "float64", "tt", 40, 6, 7279UL, false, "float64", "sha:c5f7bd8e3c7b0c7a287b0b26920933a8")]
        [DataRow("float64", "float32", "c", 1, 6, 7280UL, false, "float64", "sha:819256c15a557f274552f57fcd430f09")]
        [DataRow("float64", "float32", "rev", 1, 6, 7281UL, false, "float64", "sha:3525a849079b42ae65e73f0a8420db25")]
        [DataRow("float64", "float32", "b2", 1, 6, 7282UL, false, "float64", "sha:debb16fdbcb27c6875e48115c63ece3c")]
        [DataRow("float64", "float32", "t1", 1, 6, 7283UL, false, "float64", "sha:172456e3f8983fb5304f939ca24aae7e")]
        [DataRow("float64", "float32", "tt", 1, 6, 7284UL, false, "float64", "sha:bdbbeca9ef84bf07f2003e4d538062fa")]
        [DataRow("float64", "float32", "c", 5, 6, 7285UL, false, "float64", "sha:b5689f4513a5822b106d9c1799a871df")]
        [DataRow("float64", "float32", "rev", 5, 6, 7286UL, false, "float64", "sha:3ceb038e26fa1f6d751124c5b3391217")]
        [DataRow("float64", "float32", "b2", 5, 6, 7287UL, false, "float64", "sha:e993b08e5f49965ae5d8566267b1e181")]
        [DataRow("float64", "float32", "t1", 5, 6, 7288UL, false, "float64", "sha:b6dca866b0643bbd2a0f0fbf21f50167")]
        [DataRow("float64", "float32", "tt", 5, 6, 7289UL, false, "float64", "sha:84d59afc2d9110e51d00b83d01284a5b")]
        [DataRow("float64", "float32", "c", 40, 6, 7290UL, false, "float64", "sha:5d1d13c3bcc2731d5923310565d4bdab")]
        [DataRow("float64", "float32", "rev", 40, 6, 7291UL, false, "float64", "sha:245728ef72571bdd5184ae196c92564c")]
        [DataRow("float64", "float32", "b2", 40, 6, 7292UL, false, "float64", "sha:daafcd8c018308f5a7f261f7c6c034fa")]
        [DataRow("float64", "float32", "t1", 40, 6, 7293UL, false, "float64", "sha:e3d50d74c169892f7bb649cfe8d2cc24")]
        [DataRow("float64", "float32", "tt", 40, 6, 7294UL, false, "float64", "sha:dda5ad55046ac2c35d810a45bb450d6f")]
        [DataRow("int32", "float32", "c", 1, 6, 7295UL, false, "float64", "sha:0b28a9548f3016ffe27612c7dd65f87b")]
        [DataRow("int32", "float32", "rev", 1, 6, 7296UL, false, "float64", "sha:a114a0ec1d5250bfadc3268b77e2dc81")]
        [DataRow("int32", "float32", "b2", 1, 6, 7297UL, false, "float64", "sha:eaa9d53b1a889b40b5ce883ce5803841")]
        [DataRow("int32", "float32", "t1", 1, 6, 7298UL, false, "float64", "sha:a958b3e9290d2e7c2e7b053b18ec11d9")]
        [DataRow("int32", "float32", "tt", 1, 6, 7299UL, false, "float64", "sha:a8609e79564c1f058060c3913a5a59c9")]
        [DataRow("int32", "float32", "c", 5, 6, 7300UL, false, "float64", "sha:eb83a9a2728373881e91abbce8200fda")]
        [DataRow("int32", "float32", "rev", 5, 6, 7301UL, false, "float64", "sha:7fd814701725d11d3063abacd2df6038")]
        [DataRow("int32", "float32", "b2", 5, 6, 7302UL, false, "float64", "sha:e47ef778d431fb4cfd61ed2aafcd1f1f")]
        [DataRow("int32", "float32", "t1", 5, 6, 7303UL, false, "float64", "sha:99c656350925470483cac19fcdc3fd8b")]
        [DataRow("int32", "float32", "tt", 5, 6, 7304UL, false, "float64", "sha:df1d24a398d4cc88b11ab3b350bf57f9")]
        [DataRow("int32", "float32", "c", 40, 6, 7305UL, false, "float64", "sha:abe1d033644042ff5ad74547d5e5774b")]
        [DataRow("int32", "float32", "rev", 40, 6, 7306UL, false, "float64", "sha:868ad389e56f9b1be7149d6b7d865a64")]
        [DataRow("int32", "float32", "b2", 40, 6, 7307UL, false, "float64", "sha:4c1db99d7a1fefbe8ed531c13f34633d")]
        [DataRow("int32", "float32", "t1", 40, 6, 7308UL, false, "float64", "sha:6bc51c02ebe6eeb2252ff15be25c668a")]
        [DataRow("int32", "float32", "tt", 40, 6, 7309UL, false, "float64", "sha:794825fc94b0bee7163759d1025beb74")]
        [DataRow("float16", "float32", "c", 1, 6, 7310UL, false, "float32", "sha:b1c42715238633d1ec7517d9ca428833")]
        [DataRow("float16", "float32", "rev", 1, 6, 7311UL, false, "float32", "sha:104d24dca0c1dcb174988d4394d25098")]
        [DataRow("float16", "float32", "b2", 1, 6, 7312UL, false, "float32", "sha:9316aa9c5e8f764b0b9c95e1fd1657d8")]
        [DataRow("float16", "float32", "t1", 1, 6, 7313UL, false, "float32", "sha:d204b619d146c0be240bd26f402df85a")]
        [DataRow("float16", "float32", "tt", 1, 6, 7314UL, false, "float32", "sha:ca838dbd6d2db1eba02b2cd6c60a6373")]
        [DataRow("float16", "float32", "c", 5, 6, 7315UL, false, "float32", "sha:00c4f18db5bba27806f7ab30c8e469bf")]
        [DataRow("float16", "float32", "rev", 5, 6, 7316UL, false, "float32", "sha:b0e1283b77daa6f27292cb6adbaf6a73")]
        [DataRow("float16", "float32", "b2", 5, 6, 7317UL, false, "float32", "sha:18bf521c106bb12f332cb732e633c16a")]
        [DataRow("float16", "float32", "t1", 5, 6, 7318UL, false, "float32", "sha:6a9e32d548f84a6fbe7623bac590bcde")]
        [DataRow("float16", "float32", "tt", 5, 6, 7319UL, false, "float32", "sha:c66af37fa9be47d75c33b433efeca596")]
        [DataRow("float16", "float32", "c", 40, 6, 7320UL, false, "float32", "sha:fd4fdb59d1762b3f36859a286e5d818f")]
        [DataRow("float16", "float32", "rev", 40, 6, 7321UL, false, "float32", "sha:93d637e62bd82241ab5963c45bce2311")]
        [DataRow("float16", "float32", "b2", 40, 6, 7322UL, false, "float32", "sha:65a4d0a34aca5af3c09a6ecf3173d01d")]
        [DataRow("float16", "float32", "t1", 40, 6, 7323UL, false, "float32", "sha:49b49c5154e427ab4028f4de087eec92")]
        [DataRow("float16", "float32", "tt", 40, 6, 7324UL, false, "float32", "sha:3132c29456a966e6c8762b58306e2804")]
        [DataRow("bool", "float64", "c", 1, 6, 7325UL, false, "float64", "sha:dd4857054664c2840c5ae202234fee97")]
        [DataRow("bool", "float64", "rev", 1, 6, 7326UL, false, "float64", "sha:2fcc44a7f0c9b161c6c9f12a8b9152dc")]
        [DataRow("bool", "float64", "b2", 1, 6, 7327UL, false, "float64", "sha:19649b1f5a8b8237eda60c6e380feb9b")]
        [DataRow("bool", "float64", "t1", 1, 6, 7328UL, false, "float64", "sha:a1eadcbc7ab7d70550e2ee9bcce1c6ae")]
        [DataRow("bool", "float64", "tt", 1, 6, 7329UL, false, "float64", "sha:834daf4ba433cdb40a8a65b24141cc90")]
        [DataRow("bool", "float64", "c", 5, 6, 7330UL, false, "float64", "sha:8efac7438886466b8dba29c7119ee34f")]
        [DataRow("bool", "float64", "rev", 5, 6, 7331UL, false, "float64", "sha:6463d40081fd915c9ebd7d34ec64f149")]
        [DataRow("bool", "float64", "b2", 5, 6, 7332UL, false, "float64", "sha:b41b2229fedf32e3442b69671407282f")]
        [DataRow("bool", "float64", "t1", 5, 6, 7333UL, false, "float64", "sha:7e9aa2ea271b75c310ef253e19f7c44d")]
        [DataRow("bool", "float64", "tt", 5, 6, 7334UL, false, "float64", "sha:10244ffc1ce5a7757744521ab906e8eb")]
        [DataRow("bool", "float64", "c", 40, 6, 7335UL, false, "float64", "sha:ab024ddf9851db06df5cbba0d53e99e4")]
        [DataRow("bool", "float64", "rev", 40, 6, 7336UL, false, "float64", "sha:02ec66f8423fbe0809520ac25ac8c0e6")]
        [DataRow("bool", "float64", "b2", 40, 6, 7337UL, false, "float64", "sha:2cf5f6481e1f052a3167e63e699ee9ef")]
        [DataRow("bool", "float64", "t1", 40, 6, 7338UL, false, "float64", "sha:e34ff2602f7850ee58b7a35ff8bd9143")]
        [DataRow("bool", "float64", "tt", 40, 6, 7339UL, false, "float64", "sha:3c044619fcd265c41d3a577b00a43644")]
        [DataRow("float64", "complex128", "c", 1, 6, 7340UL, false, "complex128", "sha:e3bbf38e53b8f0d8078fa4594d9067b6")]
        [DataRow("float64", "complex128", "rev", 1, 6, 7341UL, false, "complex128", "sha:87dee8081e52819651c8fadd3498917d")]
        [DataRow("float64", "complex128", "b2", 1, 6, 7342UL, false, "complex128", "sha:1b4ed7b1ddf934ca53124abc07254cf2")]
        [DataRow("float64", "complex128", "t1", 1, 6, 7343UL, false, "complex128", "sha:0b0d05ab3ce23e97ef28ee753116c4bb")]
        [DataRow("float64", "complex128", "tt", 1, 6, 7344UL, false, "complex128", "sha:213836e61301bbd26f95c64a4d50a8c4")]
        [DataRow("float64", "complex128", "c", 5, 6, 7345UL, false, "complex128", "sha:85b6b815658f176b2775efd68482d5fc")]
        [DataRow("float64", "complex128", "rev", 5, 6, 7346UL, false, "complex128", "sha:2c45de561f10615e9bfb0120375dc700")]
        [DataRow("float64", "complex128", "b2", 5, 6, 7347UL, false, "complex128", "sha:bee7ccb8f91d473fe00f3d61da7618cd")]
        [DataRow("float64", "complex128", "t1", 5, 6, 7348UL, false, "complex128", "sha:23d29add6adca016817ff06115bac49a")]
        [DataRow("float64", "complex128", "tt", 5, 6, 7349UL, false, "complex128", "sha:cb027777af4974822f0e6d64d199eb04")]
        [DataRow("float64", "complex128", "c", 40, 6, 7350UL, false, "complex128", "sha:5c1811772bbaec877afd0d3fbff00bd7")]
        [DataRow("float64", "complex128", "rev", 40, 6, 7351UL, false, "complex128", "sha:87259ea6c20697efbc0d9ec763a05b24")]
        [DataRow("float64", "complex128", "b2", 40, 6, 7352UL, false, "complex128", "sha:ed8d0fd768127989d3ef725a53be54ee")]
        [DataRow("float64", "complex128", "t1", 40, 6, 7353UL, false, "complex128", "sha:dcefff3ccfc4a415716a28b50faebf88")]
        [DataRow("float64", "complex128", "tt", 40, 6, 7354UL, false, "complex128", "sha:41a937b3752884d71fa56777ef1b200a")]
        [DataRow("int64", "complex128", "c", 1, 6, 7355UL, false, "complex128", "sha:0fb9dddfd339a21a3f155d1e0c5b1b69")]
        [DataRow("int64", "complex128", "rev", 1, 6, 7356UL, false, "complex128", "sha:db77fb94d52c4a2e1b8da85f87ea21ac")]
        [DataRow("int64", "complex128", "b2", 1, 6, 7357UL, false, "complex128", "sha:eaf387e8ea49ee55fdc873c1aadefa64")]
        [DataRow("int64", "complex128", "t1", 1, 6, 7358UL, false, "complex128", "sha:079108876ada686c2177472ee9276f6d")]
        [DataRow("int64", "complex128", "tt", 1, 6, 7359UL, false, "complex128", "sha:c23c752f416348573faf563385c84353")]
        [DataRow("int64", "complex128", "c", 5, 6, 7360UL, false, "complex128", "sha:7f33910ddc2641b113f423314461b8bf")]
        [DataRow("int64", "complex128", "rev", 5, 6, 7361UL, false, "complex128", "sha:dd585ae11324271cbf2eea7baa9b6123")]
        [DataRow("int64", "complex128", "b2", 5, 6, 7362UL, false, "complex128", "sha:c88cb62a739fa5542fb60f9bc9844804")]
        [DataRow("int64", "complex128", "t1", 5, 6, 7363UL, false, "complex128", "sha:3756f89d1b17b78bef5ba8b7271ed824")]
        [DataRow("int64", "complex128", "tt", 5, 6, 7364UL, false, "complex128", "sha:cfa4d7ad08a0345c2ceaa261ed61b5bb")]
        [DataRow("int64", "complex128", "c", 40, 6, 7365UL, false, "complex128", "sha:66814aa004a5082904d9caa1c604bcb6")]
        [DataRow("int64", "complex128", "rev", 40, 6, 7366UL, false, "complex128", "sha:3399b7529170bc8b735d16a87ddd6ea5")]
        [DataRow("int64", "complex128", "b2", 40, 6, 7367UL, false, "complex128", "sha:17af849cfc646b8a15b5dc7d85f804ca")]
        [DataRow("int64", "complex128", "t1", 40, 6, 7368UL, false, "complex128", "sha:46e108cb71eed66f73456123d1b668d1")]
        [DataRow("int64", "complex128", "tt", 40, 6, 7369UL, false, "complex128", "sha:21a0b4ec87381b60565d6fb0766a2cd9")]
        [DataRow("float32", "float64", "c", 3, 700, 7370UL, false, "float64", "sha:f20dfbd0a87ed5ecad3af05f0c55c543")]
        [DataRow("float32", "float64", "rev", 3, 700, 7371UL, false, "float64", "sha:10f5df68081f3a90ef73fe0983ca99a0")]
        [DataRow("int32", "float32", "c", 3, 1500, 7372UL, false, "float64", "sha:503a924fedc1a680c9c41cd8bce861cb")]
        [DataRow("int32", "float32", "rev", 3, 1500, 7373UL, false, "float64", "sha:b948a60a0d2dc136dba482b8153c8faa")]
        [DataRow("float64", "complex128", "c", 3, 400, 7374UL, false, "complex128", "sha:ec2fb15525b2bd3d15ee794981c235e6")]
        [DataRow("float64", "complex128", "rev", 3, 400, 7375UL, false, "complex128", "sha:b93590dbf70cb2c31de5212a135795fd")]
        [DataRow("float32", "float64", "c", 1, 2100, 7376UL, false, "float64", "sha:52dc498b2c79280eaeb543e01e1893fc")]
        [DataRow("float32", "float64", "rev", 1, 2100, 7377UL, false, "float64", "sha:7f57181fc9cc526ec6b93b08e2e14bfe")]
        [DataRow("float64", "complex128", "c", 1, 1100, 7378UL, false, "complex128", "sha:f0a370ec5b08d00c27fa422b66145699")]
        [DataRow("float64", "complex128", "rev", 1, 1100, 7379UL, false, "complex128", "sha:02a8b1a4b41840ca92149bc0e1ddaa8f")]
        public void Vecdot_MatchesNumPyBits(string dtypeA, string dtypeB, string kind, int n, int m, ulong seed,
            bool special, string resultDtype, string expected)
        {
            var g = new Lcg(seed);
            var (sa, sb) = kind switch
            {
                "b2" => (new long[] { m, n }, new long[] { n }),
                "t1" => (new long[] { n, m }, new long[] { m, n }),
                "tt" => (new long[] { n, m }, new long[] { n, m }),
                "step" => (new long[] { m, 2L * n }, new long[] { m, 2L * n }),
                "bcore" => (new long[] { m, n }, new long[] { m, 1 }),
                "d3" => (new long[] { 2, 1, 3, n }, new long[] { 4, 1, n }),
                _ => (new long[] { m, n }, new long[] { m, n }),
            };
            var a = g.Array(sa, dtypeA);
            var b = g.Array(sb, dtypeB);
            if (special)
            {
                Specials(a);
                if (kind != "bcore" && b.size > 2)   // a broadcast core, or a one-element b, keeps its draws
                    Specials(b);
            }
            var (av, bv) = kind switch
            {
                "rev" => (a[":, ::-1"], b),
                "t1" => (a.T, b),
                "tt" => (a.T, b.T),
                "step" => (a[":, ::2"], b[":, ::2"]),
                "bcore" => (a, np.broadcast_to(b, new Shape(m, n))),
                _ => (a, b),
            };

            AssertNumPyBits(np.vecdot(av, bv), resultDtype, expected, $"vecdot {dtypeA}.{dtypeB} {kind} n={n} m={m}");
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
                    .Should().Be(expected.Substring(4), "the result must be NumPy's bits for {0}", what);
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
