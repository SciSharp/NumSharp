using System;
using System.Linq;
using System.Runtime.Intrinsics;
using NumSharp.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// The float32 unary loops NumSharp ports from NumPy 2.4.2's OWN kernels rather than delegating
    /// to the platform libm — <c>log</c> (<c>simd_log_FLOAT</c>), <c>sin</c>/<c>cos</c>
    /// (<c>simd_sincos_f32</c>) — plus <c>rad2deg</c>/<c>deg2rad</c>, whose constant NumPy forms at
    /// the operand's own precision. Companion to <see cref="NpExpFloat32ParityTest"/>, which covers
    /// <c>exp</c>.
    ///
    /// <para>Every expectation is a raw bit pattern probed from NumPy 2.4.2 — a tolerance-based
    /// assertion would pass on the OLD implementations too, which is exactly the failure mode this
    /// work removes. Each of these kernels is verified exhaustively (all 2^32 float32 inputs, both
    /// the scalar and vector paths) during development; the <c>numpy_f32_kernels.jsonl</c> fuzz tier
    /// pins the discriminating inputs in CI.</para>
    /// </summary>
    [TestClass]
    public class NumPyFloat32KernelsTest : TestClass
    {
        private static float F(uint bits) => BitConverter.UInt32BitsToSingle(bits);
        private static uint B(float v) => BitConverter.SingleToUInt32Bits(v);

        // ---------------------------------------------------------------- log

        /// <summary>(input, NumPy's log output, what it pins) — probed from NumPy 2.4.2.</summary>
        private static readonly (uint In, uint Out, string What)[] LogProbed =
        {
            // A NaN argument yields the POSITIVE canonical NaN, but a NEGATIVE argument yields a
            // NEGATIVE one (NumPy stores -NPY_NANF there). A correctly-rounded libm has no such
            // asymmetry, so these two lines alone separate the port from MathF.Log.
            (0x7fc00000u, 0x7fc00000u, "log(NaN) = +NaN"),
            (0xffc00000u, 0x7fc00000u, "log(-NaN) = +NaN (the NaN test wins over the sign test)"),
            (0xbf800000u, 0xffc00000u, "log(-1) = -NaN"),
            (0xff800000u, 0xffc00000u, "log(-inf) = -NaN"),

            (0x7f800000u, 0x7f800000u, "log(+inf) = +inf"),
            (0x00000000u, 0xff800000u, "log(+0) = -inf"),
            (0x80000000u, 0xff800000u, "log(-0) = -inf (ordered compare, so -0 counts as zero)"),

            (0x3f800000u, 0x00000000u, "log(1) = +0"),
            (0x402df854u, 0x3f800000u, "log(e) = 1"),
            (0x3f000000u, 0xbf317218u, "log(0.5)"),
            (0x00000001u, 0xc2ce8ed0u, "smallest subnormal — exercises the 2^100 rescale"),
            (0x3f3504f3u, 0xbeb17218u, "1/sqrt(2), the mantissa split point"),
            (0x3f486945u, 0xbe7aae7au, "NumPy's documented worst case (3.83 ULP)"),
            (0x7f7fffffu, 0x42b17218u, "max finite"),
        };

        [TestMethod]
        public void Log_Float32_MatchesNumPyBitForBit()
        {
            var result = np.log(np.array(LogProbed.Select(p => F(p.In)).ToArray()));
            result.dtype.Should().Be(typeof(float));
            for (int i = 0; i < LogProbed.Length; i++)
                B(result.GetSingle(i)).Should().Be(LogProbed[i].Out, LogProbed[i].What);
        }

        // ------------------------------------------------------------ sin/cos

        private static readonly (uint In, uint Out, string What)[] SinProbed =
        {
            (0x7fc00000u, 0x7fc00000u, "sin(NaN) = canonical NaN"),
            (0x7f800000u, 0xffc00000u, "sin(+inf) = -NaN (from the libc fallback)"),
            (0xff800000u, 0xffc00000u, "sin(-inf) = -NaN"),
            (0x00000000u, 0x00000000u, "sin(+0) = +0"),
            (0x80000000u, 0x80000000u, "sin(-0) = -0 — the sign survives the reduction"),
            (0x3fc90fdbu, 0x3f800000u, "sin(pi/2) = 1"),
            (0x40490fdbu, 0xb3bbbd2eu, "sin(pi) — the residual of the float pi"),
            (0x3f800000u, 0x3f576aa5u, "sin(1)"),
            // NumPy hands arguments past 117435.992 to libc; these straddle that seam.
            (0x47e55dffu, 0xbdef7be3u, "sin at the Cody-Waite limit (last polynomial input)"),
            (0x47e55e00u, 0xbdff5ddfu, "sin one ULP past it (first libc input)"),
            (0x4b000000u, 0x3edd4fa3u, "sin far past the limit"),
        };

        private static readonly (uint In, uint Out, string What)[] CosProbed =
        {
            (0x7fc00000u, 0x7fc00000u, "cos(NaN) = canonical NaN"),
            (0x7f800000u, 0xffc00000u, "cos(+inf) = -NaN"),
            (0x00000000u, 0x3f800000u, "cos(+0) = 1"),
            (0x40490fdbu, 0xbf800000u, "cos(pi) = -1"),
            (0x40000000u, 0xbed51132u, "cos(2)"),
            // Cosine's libc cutoff is a DIFFERENT (smaller) constant than sine's.
            (0x478b9a08u, 0x3def8fc6u, "cos at ITS Cody-Waite limit, 71476.0625"),
            (0x478b9a09u, 0x3dff71bdu, "cos one ULP past it"),
        };

        // NumPy hands BOTH ±inf AND finite inputs past the kernel's Cody-Waite reduction limit to
        // the PLATFORM libm, exactly as NumSharp does, so the result is host-specific: macOS libm
        // returns a POSITIVE NaN for sin(±inf) where NumPy's win-amd64 wheel (MSVC ucrtbase) returns
        // a NEGATIVE one, and the last bit of a finite past-limit result differs by ~1 ULP on
        // glibc/macOS. NumPy's expected bytes here were probed from that MSVC wheel, so pin these
        // libm-fallback cases on Windows only. A NaN INPUT is blanked by NumSharp's own kernel
        // (portable), and the polynomial range below the limit is NumSharp's own port (portable), so
        // neither is skipped — both keep asserting on every OS.
        private static bool HostLibmPast(uint inBits, uint firstLibmBits)
        {
            if (OperatingSystem.IsWindows())
                return false;
            float x = F(inBits);
            return float.IsInfinity(x) || (float.IsFinite(x) && MathF.Abs(x) >= F(firstLibmBits));
        }

        [TestMethod]
        public void Sin_Float32_MatchesNumPyBitForBit()
        {
            var result = np.sin(np.array(SinProbed.Select(p => F(p.In)).ToArray()));
            result.dtype.Should().Be(typeof(float));
            for (int i = 0; i < SinProbed.Length; i++)
            {
                if (HostLibmPast(SinProbed[i].In, 0x47e55e00u))   // sine's first libc input
                    continue;
                B(result.GetSingle(i)).Should().Be(SinProbed[i].Out, SinProbed[i].What);
            }
        }

        [TestMethod]
        public void Cos_Float32_MatchesNumPyBitForBit()
        {
            var result = np.cos(np.array(CosProbed.Select(p => F(p.In)).ToArray()));
            result.dtype.Should().Be(typeof(float));
            for (int i = 0; i < CosProbed.Length; i++)
            {
                if (HostLibmPast(CosProbed[i].In, 0x478b9a09u))   // cosine's first libc input
                    continue;
                B(result.GetSingle(i)).Should().Be(CosProbed[i].Out, CosProbed[i].What);
            }
        }

        // -------------------------------------------------------- rad2deg/deg2rad

        [TestMethod]
        public void Rad2Deg_Float32_UsesNumPysFloatPrecisionConstant()
        {
            // NumPy's RAD2DEG is (180/PI) evaluated at the OPERAND's precision. Rounding the double
            // quotient to float gives 0x42652ee1 — one ULP high — which shifted 79% of all float32
            // rad2deg results. The value below IS that discriminator: rad2deg(1).
            var probed = new (uint In, uint Deg, uint Rad)[]
            {
                (0x3f800000u, 0x42652ee0u, 0x3c8efa35u),   // 1
                (0x40490fdbu, 0x43340000u, 0x3d60969du),   // pi
                (0xbf800000u, 0xc2652ee0u, 0xbc8efa35u),   // -1
                (0x00000001u, 0x00000039u, 0x00000000u),   // subnormal: scales up / flushes to zero
                (0x501502f9u, 0x530566f2u, 0x4d267294u),   // 1e10
            };
            var input = np.array(probed.Select(p => F(p.In)).ToArray());

            var deg = np.rad2deg(input);
            var rad = np.deg2rad(input);
            for (int i = 0; i < probed.Length; i++)
            {
                B(deg.GetSingle(i)).Should().Be(probed[i].Deg, $"rad2deg(0x{probed[i].In:x8})");
                B(rad.GetSingle(i)).Should().Be(probed[i].Rad, $"deg2rad(0x{probed[i].In:x8})");
            }
        }

        [TestMethod]
        public void Rad2Deg_Float32_ContiguousAndStridedAgree()
        {
            // The scalar emitter and the vector emitter carry the constant SEPARATELY; if they ever
            // drift apart, a contiguous array and a strided view of it would disagree.
            var xs = Enumerable.Range(0, 64).Select(i => (float)(i * 0.37 - 12.0)).ToArray();
            var flat = np.array(xs);
            var contiguous = np.rad2deg(flat);
            var strided = np.rad2deg(flat["::2"]);
            for (int i = 0; i < xs.Length / 2; i++)
                B(strided.GetSingle(i)).Should().Be(B(contiguous.GetSingle(i * 2)),
                    "the vector and scalar rad2deg factors must be the same float");
        }

        // ------------------------------------------------- scalar vs vector agreement

        [TestMethod]
        public void PortedKernels_VectorOverloads_AgreeWithScalar_AtEveryWidth()
        {
            // Vector512/128 may not be this host's active width, and Avx512F may be absent, but the
            // overloads stay callable (the FMA falls back to MathF.FusedMultiplyAdd), so all three
            // widths are exercised wherever the tests run.
            var xs = LogProbed.Select(p => F(p.In))
                .Concat(SinProbed.Select(p => F(p.In)))
                .Concat(CosProbed.Select(p => F(p.In)))
                .Concat(Enumerable.Range(0, 96).Select(i => (float)(i * 2.75 - 130.0)))
                .ToArray();

            for (int i = 0; i + 16 <= xs.Length; i += 16)
            {
                var v128 = Vector128.Create(xs.AsSpan(i, 4));
                var v256 = Vector256.Create(xs.AsSpan(i, 8));
                var v512 = Vector512.Create(xs.AsSpan(i, 16));

                for (int k = 0; k < 4; k++)
                {
                    B(NDFloatMath.Log(v128)[k]).Should().Be(B(NDFloatMath.Log(xs[i + k])), $"log V128 lane {k}");
                    B(NDFloatMath.Sin(v128)[k]).Should().Be(B(NDFloatMath.Sin(xs[i + k])), $"sin V128 lane {k}");
                    B(NDFloatMath.Cos(v128)[k]).Should().Be(B(NDFloatMath.Cos(xs[i + k])), $"cos V128 lane {k}");
                }
                for (int k = 0; k < 8; k++)
                {
                    B(NDFloatMath.Log(v256)[k]).Should().Be(B(NDFloatMath.Log(xs[i + k])), $"log V256 lane {k}");
                    B(NDFloatMath.Sin(v256)[k]).Should().Be(B(NDFloatMath.Sin(xs[i + k])), $"sin V256 lane {k}");
                    B(NDFloatMath.Cos(v256)[k]).Should().Be(B(NDFloatMath.Cos(xs[i + k])), $"cos V256 lane {k}");
                }
                for (int k = 0; k < 16; k++)
                {
                    B(NDFloatMath.Log(v512)[k]).Should().Be(B(NDFloatMath.Log(xs[i + k])), $"log V512 lane {k}");
                    B(NDFloatMath.Sin(v512)[k]).Should().Be(B(NDFloatMath.Sin(xs[i + k])), $"sin V512 lane {k}");
                    B(NDFloatMath.Cos(v512)[k]).Should().Be(B(NDFloatMath.Cos(xs[i + k])), $"cos V512 lane {k}");
                }
            }
        }

        [TestMethod]
        public void PortedKernels_SameBitsThroughEveryLayout()
        {
            var xs = Enumerable.Range(0, 128).Select(i => (float)(i * 0.9 - 50.0)).ToArray();
            var flat = np.array(xs);

            foreach (var (name, f) in new (string, Func<NDArray, NDArray>)[]
                     { ("log", a => np.log(a)), ("sin", a => np.sin(a)), ("cos", a => np.cos(a)),
                       ("rad2deg", a => np.rad2deg(a)) })
            {
                var expected = f(flat);
                var strided = f(flat["::2"]);
                var reversed = f(flat["::-1"]);
                var offset = f(flat["5:"]);
                for (int i = 0; i < xs.Length / 2; i++)
                    B(strided.GetSingle(i)).Should().Be(B(expected.GetSingle(i * 2)), $"{name} strided {i}");
                for (int i = 0; i < xs.Length; i++)
                    B(reversed.GetSingle(i)).Should().Be(B(expected.GetSingle(xs.Length - 1 - i)), $"{name} reversed {i}");
                for (int i = 0; i < xs.Length - 5; i++)
                    B(offset.GetSingle(i)).Should().Be(B(expected.GetSingle(i + 5)), $"{name} offset {i}");
            }
        }

        [TestMethod]
        public void PortedKernels_NarrowIntegerInputs_UseTheSameFloat32Loop()
        {
            // int16/uint16 promote to NumPy's 'f->f' loop — the very kernels ported here.
            var i16 = np.array(new short[] { 1, 2, 3, 10, 100, 1000 });
            np.log(i16).dtype.Should().Be(typeof(float));
            np.sin(i16).dtype.Should().Be(typeof(float));

            var asFloat = np.array(new float[] { 1, 2, 3, 10, 100, 1000 });
            for (int i = 0; i < 6; i++)
            {
                B(np.log(i16).GetSingle(i)).Should().Be(B(np.log(asFloat).GetSingle(i)), $"log int16 {i}");
                B(np.sin(i16).GetSingle(i)).Should().Be(B(np.sin(asFloat).GetSingle(i)), $"sin int16 {i}");
                B(np.cos(i16).GetSingle(i)).Should().Be(B(np.cos(asFloat).GetSingle(i)), $"cos int16 {i}");
            }
        }

        [TestMethod]
        public void PortedKernels_Float64_IsUnaffected()
        {
            // The ports are scoped to the float32 loops; float64 still runs Math.*, which already
            // agrees with NumPy's scalar npy_* on this platform.
            np.log(np.array(new[] { 2.0 })).GetDouble(0).Should().Be(System.Math.Log(2.0));
            np.sin(np.array(new[] { 2.0 })).GetDouble(0).Should().Be(System.Math.Sin(2.0));
            np.cos(np.array(new[] { 2.0 })).GetDouble(0).Should().Be(System.Math.Cos(2.0));
        }
    }
}
