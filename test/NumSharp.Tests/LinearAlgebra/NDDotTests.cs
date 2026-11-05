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

            /// <summary>An array of <paramref name="dtype"/> filled in C order (complex: real draw, then imaginary).
            ///     float32 values are the doubles rounded to float32, as NumPy's astype rounds them.</summary>
            /// <param name="shape">The shape.</param>
            /// <param name="dtype">float32, float64 or complex128.</param>
            /// <returns>The fresh C-contiguous array.</returns>
            public NDArray Array(long[] shape, string dtype)
            {
                long n = 1;
                foreach (long d in shape)
                    n *= d;
                switch (dtype)
                {
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

            var r = np.dot(av, bv);

            r.dtype.name.Should().Be(resultDtype);
            byte[] bytes = r.tobytes();
            if (expected.StartsWith("sha:", StringComparison.Ordinal))
            {
                Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant().Substring(0, 32)
                    .Should().Be(expected.Substring(4), "np.dot must give NumPy's bits for {0} {1} K={2} M={3}", dtypeA,
                        kind, K, M);
                return;
            }

            // Special rows: every byte, except that a NaN slot only has to be NaN.
            byte[] want = Convert.FromHexString(expected.Substring(4));
            bytes.Length.Should().Be(want.Length);
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
                    gotNaN.Should().BeTrue("slot {0} of {1} {2}: NumPy gives NaN", i / part, dtypeA, kind);
                    continue;
                }
                bytes.AsSpan(i, part).SequenceEqual(want.AsSpan(i, part)).Should().BeTrue(
                    "slot {0} of {1} {2}: NumPy {3}, NumSharp {4}", i / part, dtypeA, kind,
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
