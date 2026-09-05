using System;
using System.Linq;
using NumSharp.Interop.OpenBLAS;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Regressions pinned by the 0.70.0 release QA (2026-09-05): every case here was found by consuming the
    ///     freshly packed NuGet packages from a real consumer project against NumPy 2.4.2 output and was WRONG
    ///     at 05884375. Each test names the consumer-visible symptom it guards.
    /// </summary>
    [TestClass]
    public class Release070ConsumerQaTests
    {
        // ---------------------------------------------------------------- np.frombuffer(bytes, "float64")

        [TestMethod]
        public void frombuffer_accepts_numpy_dtype_names_like_np_dtype_does()
        {
            // NumPy's most common spelling threw NotSupportedException: only the sized codes ("<f8") parsed.
            var bytes = BitConverter.GetBytes(1.5).Concat(BitConverter.GetBytes(2.5)).ToArray();
            foreach (var spelling in new[] { "float64", "<f8", "f8", "d" })
            {
                var a = np.frombuffer(bytes, spelling);
                a.typecode.Should().Be(NPTypeCode.Double, spelling);
                a.GetDouble(1).Should().Be(2.5, spelling);
            }
            np.frombuffer(new byte[] { 1, 0, 0, 0 }, "int32").GetInt32(0).Should().Be(1);
            np.frombuffer(new byte[] { 1, 0 }, "bool").typecode.Should().Be(NPTypeCode.Boolean);
            np.frombuffer(new byte[16], "complex128").typecode.Should().Be(NPTypeCode.Complex);
            np.frombuffer(new byte[2], "float16").typecode.Should().Be(NPTypeCode.Half);
            // Unknown strings keep the verbatim rejection; complex64 keeps np.dtype's own refusal.
            Assert.ThrowsException<NotSupportedException>(() => np.frombuffer(bytes, "quaternion")).Message.Should().Contain("dtype string 'quaternion' is not supported");
            Assert.ThrowsException<NotSupportedException>(() => np.frombuffer(bytes, "complex64"));
        }

        // ---------------------------------------------------------------- DType.name

        [TestMethod]
        public void dtype_name_is_numpys_name_not_the_clr_type_name()
        {
            np.dtype("f4").name.Should().Be("float32");
            np.dtype("float64").name.Should().Be("float64");
            np.dtype("<i8").name.Should().Be("int64");
            DType.Boolean.name.Should().Be("bool");
            DType.Complex.name.Should().Be("complex128");
            DType.Half.name.Should().Be("float16");
            ((DType)typeof(byte)).name.Should().Be("uint8");
            np.dtype("f4").ToString().Should().Be("float32");
            // The NumSharp-only dtypes get their own lowercase names, not a misleading NumPy stand-in.
            DType.Char.name.Should().Be("char");
            DType.Decimal.name.Should().Be("decimal");
        }

        // ---------------------------------------------------------------- NEP50 weak int literals on float arrays

        [TestMethod]
        public void clip_with_weak_int_bounds_keeps_the_float_dtype()
        {
            var f32 = np.arange(4).astype(np.float32);
            var f16 = np.arange(4).astype(np.float16);
            np.clip(f32, 1, 3).typecode.Should().Be(NPTypeCode.Single, "np.clip(float32, 1, 3) is float32 in NumPy 2.4.2");
            np.clip(f32, 1, null).typecode.Should().Be(NPTypeCode.Single);
            np.clip(f32, null, 3).typecode.Should().Be(NPTypeCode.Single);
            np.clip(f16, 1, 3).typecode.Should().Be(NPTypeCode.Half);
            f32.clip(0, 1).typecode.Should().Be(NPTypeCode.Single);
            np.clip(f32, 1, 3).Should().BeOfValues(1f, 1f, 2f, 3f);
            // Higher-kind scalar bounds still promote, exactly as NumPy's Python float on an int array does.
            np.clip(np.arange(4), 1.0, 3.0).typecode.Should().Be(NPTypeCode.Double);
            np.clip(np.arange(4), 1, 3).typecode.Should().Be(NPTypeCode.Int64);   // np.arange is int64
            np.clip(np.arange(4).astype(np.uint8), 1, 3).typecode.Should().Be(NPTypeCode.Byte);
            // Array bounds are strong operands: an int32 ARRAY bound on float32 promotes to float64 (NumPy result_type).
            np.clip(f32, np.array(new[] { 1, 1, 1, 1 }), np.array(new[] { 3, 3, 3, 3 })).typecode.Should().Be(NPTypeCode.Double);
        }

        [TestMethod]
        public void arctan2_template_family_keeps_the_float_dtype_for_weak_int_scalars()
        {
            var f32 = np.arange(1, 5).astype(np.float32);
            var f16 = np.arange(1, 5).astype(np.float16);
            np.arctan2(f32, 1).typecode.Should().Be(NPTypeCode.Single, "np.arctan2(float32, 1) is float32 in NumPy 2.4.2");
            np.arctan2(1, f32).typecode.Should().Be(NPTypeCode.Single);
            np.arctan2(f16, 1).typecode.Should().Be(NPTypeCode.Half);
            np.copysign(f32, 1).typecode.Should().Be(NPTypeCode.Single);
            np.logaddexp(f32, 1).typecode.Should().Be(NPTypeCode.Single);
            np.logaddexp2(f32, 1).typecode.Should().Be(NPTypeCode.Single);
            np.nextafter(f32, 1).typecode.Should().Be(NPTypeCode.Single);
            // A weak FLOAT literal on a float32 array also keeps float32; on an int array it is float64.
            np.arctan2(f32, 1.5).typecode.Should().Be(NPTypeCode.Single);
            np.arctan2(np.arange(4), 1.5).typecode.Should().Be(NPTypeCode.Double);
            // Two int arrays: the float tier NumPy's loops give (int8 -> float16, int32 -> float64).
            np.arctan2(np.array(new sbyte[] { 1, 2 }), np.array(new sbyte[] { 3, 4 })).typecode.Should().Be(NPTypeCode.Half);
            np.arctan2(np.arange(4), np.arange(4)).typecode.Should().Be(NPTypeCode.Double);
            // Values are unchanged by the promotion fix.
            np.arctan2(f32, 1).GetSingle(0).Should().Be(MathF.Atan2(1f, 1f));
            np.copysign(f32, -1).GetSingle(2).Should().Be(-3f);
        }

        // ---------------------------------------------------------------- poly1d evaluation

        [TestMethod]
        public void poly1d_call_evaluates_like_numpys_p_of_x()
        {
            using var p = new poly1d(np.array(new[] { 2.0, -3.0, 0.5, 4.0 }));
            var x = np.array(new[] { -1.0, 0.0, 2.0 });
            var viaCall = p.Call(x);
            viaCall.Should().BeOfValues(-1.5, 4.0, 9.0);
            p.Call(2.0).Should().Be(9.0);
            np.polyval(p, x).Should().BeOfValues(-1.5, 4.0, 9.0);
        }

        // ---------------------------------------------------------------- OpenBLAS discovery in a single-file publish

        [TestMethod]
        public void openblas_probe_bases_include_the_runtimes_native_search_directories()
        {
            // IncludeNativeLibrariesForSelfExtract=true extracts the bundled dll into a per-app temp directory that
            // only NATIVE_DLL_SEARCH_DIRECTORIES names; the discovery must probe it or the backend silently vanishes
            // from a single-file publish.
            var bases = OpenBlasNative.ProbeBases().ToList();
            bases.Should().Contain(b => string.Equals(b.TrimEnd('\\', '/'), AppContext.BaseDirectory.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase));
            if (AppContext.GetData("NATIVE_DLL_SEARCH_DIRECTORIES") is string dirs)
                foreach (var d in dirs.Split(System.IO.Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
                    bases.Should().Contain(b => string.Equals(b.TrimEnd('\\', '/'), d.TrimEnd('\\', '/'), StringComparison.OrdinalIgnoreCase), d);
            bases.Should().OnlyHaveUniqueItems();
        }
    }
}
