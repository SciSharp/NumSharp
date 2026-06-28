using System;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.AuditV2;

/// <summary>
/// Audit V2 Tier 1 correctness bugs for Casting / Random / Utilities / Primitives.
/// Each test reproduces a verified divergence between NumSharp and NumPy 2.4.2 or
/// an internal correctness defect. Tests are kept FAILING via <see cref="OpenBugsAttribute"/>
/// so they document the bug and start passing once it is fixed.
/// </summary>
[TestClass]
public class AuditV2_CastingRandomUtilities
{
    // -----------------------------------------------------------------------
    // Helpers / fixtures
    // -----------------------------------------------------------------------

    private sealed class NpFuncTestState
    {
        public int LastResult;
        public void Op<T>(int x) where T : unmanaged
        {
            LastResult = x * 2;
        }
    }

    // -----------------------------------------------------------------------
    // T1.19 — NpFunc caches by MethodHandle.Value, ignoring instance target.
    //   File: src/NumSharp.Core/Utilities/NpFunc.cs (Resolve / ResolveSlow)
    //   The cache key is `method.Method.MethodHandle.Value` alone. When the
    //   same generic method is called against two different instance targets,
    //   the first target is bound into the cached delegate forever — every
    //   subsequent call silently dispatches against the original instance.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.19")]
    public void T1_19_NpFunc_InstanceTargetIgnored()
    {
        var s1 = new NpFuncTestState();
        var s2 = new NpFuncTestState();

        // Prime the cache against s1 — expected behaviour.
        NpFunc.Invoke(NPTypeCode.Int32, new Action<int>(s1.Op<int>), 10);
        s1.LastResult.Should().Be(20);
        s2.LastResult.Should().Be(0);

        // Second call binds to s2 — but the cache stores the delegate created
        // for s1, so s2.LastResult never changes and s1 is mutated instead.
        NpFunc.Invoke(NPTypeCode.Int32, new Action<int>(s2.Op<int>), 100);

        // NumPy reference would not have this pathology — this is a pure C# bug.
        s2.LastResult.Should().Be(200, "s2 was the explicit instance for the second invoke");
        s1.LastResult.Should().Be(20, "s1 must NOT be mutated by an invoke that bound s2");
    }

    // -----------------------------------------------------------------------
    // T1.30 — ArrayConvert inner ToX(Array) switches only handle 13/15 dtypes.
    //   File: src/NumSharp.Core/Utilities/ArrayConvert.cs
    //   Inner per-target switches lack cases for SByte/Half/Complex source
    //   arrays. Every public ArrayConvert.ToXxx(Array) throws
    //   ArgumentOutOfRangeException for these three source types.
    // -----------------------------------------------------------------------
    [TestMethod]
    public void T1_30_ArrayConvert_SByteSource_Throws()
    {
        var src = new sbyte[] { -1, 0, 1 };
        Action act = () => ArrayConvert.ToInt32(src);
        act.Should().NotThrow("SByte is one of NumSharp's 15 supported dtypes");
    }

    [TestMethod]
    public void T1_30_ArrayConvert_HalfSource_Throws()
    {
        var src = new Half[] { (Half)1.5f, (Half)2.5f };
        Action act = () => ArrayConvert.ToInt32(src);
        act.Should().NotThrow("Half is one of NumSharp's 15 supported dtypes");
    }

    [TestMethod]
    public void T1_30_ArrayConvert_ComplexSource_Throws()
    {
        var src = new Complex[] { new Complex(1, 2), new Complex(3, 0) };
        Action act = () => ArrayConvert.ToInt32(src);
        act.Should().NotThrow("Complex is one of NumSharp's 15 supported dtypes (imag silently discarded per NumPy)");
    }

    // -----------------------------------------------------------------------
    // T1.31 — `randint(low, high=-1)` uses -1 as sentinel, breaks legal call.
    //   File: src/NumSharp.Core/RandomSampling/np.random.randint.cs
    //   The default `high == -1` is the "high omitted" sentinel, then swapped
    //   to (low=0, high=low). This collides with the perfectly legal call
    //   np.random.randint(-10, -1, 3) which NumPy returns e.g. [-4, -7, -3].
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.31")]
    public void T1_31_Randint_NegativeOneHigh_TreatedAsSentinel()
    {
        np.random.seed(42);
        Action act = () => np.random.randint(-10, -1, 3);
        act.Should().NotThrow("NumPy: np.random.randint(-10, -1, 3) returns a valid array");

        // If the bug is fixed, also verify the values are in [low, high).
        np.random.seed(42);
        var arr = np.random.randint(-10, -1, 3);
        arr.size.Should().Be(3);
        for (int i = 0; i < (int)arr.size; i++)
        {
            long v = arr.GetInt64(i);
            v.Should().BeGreaterThanOrEqualTo(-10);
            v.Should().BeLessThan(-1);
        }
    }

    // -----------------------------------------------------------------------
    // T1.32 — `np.modf` tuple element name typo: "Intergral" (should be "Integral").
    //   File: src/NumSharp.Core/Math/np.modf.cs (and Backends/TensorEngine.cs +
    //         Backends/Default/Math/Default.Modf.cs).
    //   Public API typo carried throughout. Surface via TupleElementNamesAttribute
    //   on the method's return type.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.32")]
    public void T1_32_Modf_TupleElementNameTypo()
    {
        var modf = typeof(np).GetMethod(
            nameof(np.modf),
            BindingFlags.Public | BindingFlags.Static,
            null,
            new[] { typeof(NDArray), typeof(NPTypeCode?) },
            null);
        modf.Should().NotBeNull();

        // Pull the tuple element names baked in by the C# compiler.
        var attr = modf!.ReturnTypeCustomAttributes
            .GetCustomAttributes(typeof(System.Runtime.CompilerServices.TupleElementNamesAttribute), false)
            .Cast<System.Runtime.CompilerServices.TupleElementNamesAttribute>()
            .FirstOrDefault();

        attr.Should().NotBeNull("np.modf returns a named tuple");
        var names = attr!.TransformNames;
        names.Should().Contain("Fractional");
        names.Should().Contain("Integral", "the tuple element should be spelled 'Integral', not 'Intergral'");
        names.Should().NotContain("Intergral", "'Intergral' is a misspelling of 'Integral'");
    }

    // -----------------------------------------------------------------------
    // T1.33 — NPTypeCode.AsNumpyDtypeName(Char) returns "uint8" but Char is 2 bytes.
    //   File: src/NumSharp.Core/Backends/NPTypeCode.cs
    //   Char (System.Char, 2-byte UTF-16 unit) is mapped to "uint8" — the same
    //   string Byte returns. Any consumer doing dtype-name interop will see
    //   an 8-bit announcement for what is actually a 16-bit storage.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.33")]
    public void T1_33_AsNumpyDtypeName_Char_MisreportsSize()
    {
        var mi = typeof(NPTypeCodeExtensions).GetMethod(
            "AsNumpyDtypeName",
            BindingFlags.NonPublic | BindingFlags.Static);
        mi.Should().NotBeNull();

        var charName = (string)mi!.Invoke(null, new object[] { NPTypeCode.Char })!;
        var byteName = (string)mi.Invoke(null, new object[] { NPTypeCode.Byte })!;

        NPTypeCode.Char.SizeOf().Should().Be(2);
        NPTypeCode.Byte.SizeOf().Should().Be(1);

        charName.Should().NotBe(byteName,
            "Char (2 bytes) and Byte (1 byte) must not share the same numpy dtype name");
        charName.Should().NotBe("uint8",
            "Char is 2 bytes; closest NumPy analogue is 'uint16' or 'U1' (unicode), never 'uint8'");
    }

    // -----------------------------------------------------------------------
    // T1.51 — DType.byteorder is parsed but discarded. Always '='.
    //   File: src/NumSharp.Core/Creation/np.dtype.cs
    //   `np.dtype(">i4")` should expose byteorder '>' (big-endian) but NumSharp
    //   strips the prefix and unconditionally sets byteorder = '='. The
    //   constructor hardcodes byteorder = '=' regardless of the input string.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.51")]
    public void T1_51_DType_Byteorder_BigEndianPrefix_LostAsNative()
    {
        // Host machines are little-endian, so '<' and '=' are NumPy-equivalent.
        // The bug surfaces with '>' (big-endian) which must NOT collapse to '='.
        var d = np.dtype(">i4");
        d.byteorder.Should().Be('>',
            "NumPy: np.dtype('>i4').byteorder == '>' on a little-endian host");
    }

    // -----------------------------------------------------------------------
    // T1.52 — DType.kind confuses TYPECHAR with kind code.
    //   File: src/NumSharp.Core/Creation/np.dtype.cs (_kind_list_map)
    //   bool → '?' (TYPECHAR for bool, NumPy kind is 'b').
    //   Char → 'S' (S = byte string, NumPy kind for U-class is 'U').
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.52")]
    public void T1_52_DType_Kind_Bool_UsesTypecharNotKind()
    {
        // NumPy: np.dtype(bool).kind == 'b'
        var d = np.dtype("bool");
        d.kind.Should().Be('b',
            "NumPy: np.dtype(bool).kind == 'b' (kind), not '?' (TYPECHAR)");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.52")]
    public void T1_52_DType_Kind_Char_UsesStringInsteadOfUnicode()
    {
        // NumSharp's `Char` corresponds to System.Char (2-byte UTF-16 unit). The
        // closest NumPy analogue is a unicode dtype (kind 'U'), not byte-string
        // (kind 'S'). Reporting 'S' miscategorises the dtype for downstream
        // consumers that branch on kind.
        var d = np.dtype("char");
        d.kind.Should().NotBe('S',
            "Char is 2-byte UTF-16; kind 'S' (byte string) is wrong — should be 'U' (unicode)");
    }

    // -----------------------------------------------------------------------
    // T1.53 — DType.name returns C# typename, not NumPy dtype name.
    //   File: src/NumSharp.Core/Creation/np.dtype.cs (DType ctor)
    //   name = type.Name — yields "Int32", "Double", "Boolean", "Complex".
    //   NumPy: int32, float64, bool, complex128.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.53")]
    public void T1_53_DType_Name_ReturnsCSharpType_NotNumpyName()
    {
        np.dtype("int32").name.Should().Be("int32",
            "NumPy: np.dtype('int32').name == 'int32'");
        np.dtype("float64").name.Should().Be("float64",
            "NumPy: np.dtype('float64').name == 'float64'");
        np.dtype("bool").name.Should().Be("bool",
            "NumPy: np.dtype(bool).name == 'bool'");
        np.dtype("complex128").name.Should().Be("complex128",
            "NumPy: np.dtype(complex).name == 'complex128'");
    }

    // -----------------------------------------------------------------------
    // T1.58 — Default.BooleanMask fallback copies via Buffer.MemoryCopy per element.
    //   File: src/NumSharp.Core/Backends/Default/Indexing/Default.BooleanMask.cs
    //   The BooleanMaskGatherKernel.Execute inner loop invokes
    //   System.Buffer.MemoryCopy once per matched element, paying interop
    //   overhead on every gather. Test is a smoke timing sanity check, not a
    //   strict perf SLA — the gather path should be within 10x of the SIMD
    //   contiguous path on a half-true int32 mask.
    // -----------------------------------------------------------------------
    [TestMethod]
    public void T1_58_BooleanMask_Fallback_PerElement_MemoryCopy()
    {
        const int N = 100_000;
        var bigArr = np.arange(N * 2).astype(NPTypeCode.Int32);

        // Step slice forces non-contiguous (stride != elem-size) which routes
        // through BooleanMaskFallback (the path under audit).
        var strided = bigArr["::2"];
        strided.Shape.IsContiguous.Should().BeFalse();

        var maskBuf = new bool[N];
        for (int i = 0; i < N; i++)
            maskBuf[i] = (i % 2 == 0);
        var mask = np.array(maskBuf);

        var swStrided = Stopwatch.StartNew();
        int reps = 50;
        for (int i = 0; i < reps; i++)
        {
            var _ = strided[mask];
        }
        swStrided.Stop();

        // Compare against the SIMD-contiguous baseline.
        var contig = np.arange(N).astype(NPTypeCode.Int32);
        var swContig = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++)
        {
            var _ = contig[mask];
        }
        swContig.Stop();

        // Bug: fallback uses Buffer.MemoryCopy per matched element. Expect to
        // beat or match contig timing once a sane batched memcpy / typed copy
        // is used. Currently the fallback path tends to lose by >2-3x.
        swStrided.ElapsedMilliseconds.Should().BeLessThan(
            swContig.ElapsedMilliseconds * 2 + 50,
            "fallback should not be > 2x slower than SIMD baseline once Buffer.MemoryCopy" +
            " per-element is replaced by a typed loop or single block copy");
    }

    // -----------------------------------------------------------------------
    // T1.59 — np.where(condition) returns array-of-NDArray<long>; NumPy returns tuple.
    //   File: src/NumSharp.Core/APIs/np.where.cs
    //   Cosmetic API divergence — NumPy ALWAYS returns a tuple even for a
    //   single-axis condition. Treat as Misaligned API behaviour, kept as
    //   OpenBugs until either a tuple return is introduced or marked
    //   intentionally divergent.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.59")]
    public void T1_59_Where_SingleArg_ReturnsArrayNotTuple()
    {
        // Tag note: this is a known C# vs Python API mismatch. NumPy returns
        // a tuple even for 1-D conditions; NumSharp returns NDArray<long>[].
        // We assert the EXPECTED NumPy semantics (length-1 tuple) — the test
        // currently fails because NumSharp does not have a tuple wrapper.
        var c = np.array(new[] { true, false, true, false, true });
        object r = np.where(c);

        // Surrogate for "tuple-shaped" — Python tuple has length 1 here.
        // The current NumSharp return is NDArray<long>[1]. If a tuple-shaped
        // wrapper is later introduced, this assertion captures the contract.
        r.Should().BeOfType<ValueTuple<NumSharp.Generic.NDArray<long>>>(
            "NumPy returns tuple(arr,) — NumSharp currently returns NDArray<long>[]");
    }

    // -----------------------------------------------------------------------
    // T1.60 — np.where(cond, 1, 2) returns int32; NumPy returns int64.
    //   File: src/NumSharp.Core/APIs/np.where.cs / type promotion.
    //   NEP 50: Python integer literals are weak ints and resolve to platform
    //   default (int64 on 64-bit). NumSharp drops to int32 (C# default int).
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.60")]
    public void T1_60_Where_ScalarBranches_DtypePromotion()
    {
        var c = np.array(new[] { true, false, true, false });
        var r = np.where(c, (object)1, (object)2);

        r.typecode.Should().Be(NPTypeCode.Int64,
            "NumPy 2.x: np.where(cond, 1, 2).dtype == int64 (NEP 50 weak-int promotion)");
    }

    // -----------------------------------------------------------------------
    // T1.65 — NPTypeCode.Decimal → NPY_LONGLONGLTR ('q' = int64). Round-trip
    // converts Decimal -> Int64 via TYPECHAR.
    //   File: src/NumSharp.Core/Backends/NPTypeCode.cs (ToTYPECHAR + ToTypeCode)
    //   Decimal has no NumPy analogue, but mapping it to 'q' (int64) is an
    //   unsafe choice — any consumer that round-trips loses Decimal identity.
    //   Additional collateral collisions: NPY_CHARLTR == NPY_COMPLEXLTR ('c')
    //   and NPY_BYTELTR == NPY_GENBOOLLTR ('b').
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.65")]
    public void T1_65_Decimal_TypecharRoundTripLosesIdentity()
    {
        var asType = typeof(NPTypeCodeExtensions);
        var toTypechar = asType.GetMethod("ToTYPECHAR", BindingFlags.NonPublic | BindingFlags.Static);
        var toTypecode = asType.GetMethod("ToTypeCode", BindingFlags.NonPublic | BindingFlags.Static);
        toTypechar.Should().NotBeNull();
        toTypecode.Should().NotBeNull();

        var ch = toTypechar!.Invoke(null, new object[] { NPTypeCode.Decimal });
        var rt = (NPTypeCode)toTypecode!.Invoke(null, new object[] { ch! })!;

        rt.Should().Be(NPTypeCode.Decimal,
            "Decimal -> TYPECHAR -> NPTypeCode round-trip must preserve identity; currently maps to NPY_LONGLONGLTR -> Int64");
    }

    [TestMethod, OpenBugs(IssueUrl = "audit-v2-T1.65")]
    public void T1_65_Char_TypecharRoundTripLosesIdentity()
    {
        // Collateral collision: NPY_CHARLTR ('c') == NPY_COMPLEXLTR ('c').
        // Char -> 'c' -> ToTypeCode('c') hits the NPY_COMPLEXLTR case first and
        // resolves to Complex.
        var asType = typeof(NPTypeCodeExtensions);
        var toTypechar = asType.GetMethod("ToTYPECHAR", BindingFlags.NonPublic | BindingFlags.Static);
        var toTypecode = asType.GetMethod("ToTypeCode", BindingFlags.NonPublic | BindingFlags.Static);

        var ch = toTypechar!.Invoke(null, new object[] { NPTypeCode.Char });
        var rt = (NPTypeCode)toTypecode!.Invoke(null, new object[] { ch! })!;

        rt.Should().Be(NPTypeCode.Char,
            "Char -> TYPECHAR -> NPTypeCode round-trip must preserve identity (NPY_CHARLTR collides with NPY_COMPLEXLTR)");
    }

    // -----------------------------------------------------------------------
    // T1.66 — np.dtype('float') silently accepted; NumPy 2.x deprecates.
    //
    //   FALSE POSITIVE — verified against NumPy 2.4.2.
    //   `np.dtype('float')` is fully supported and emits NO DeprecationWarning
    //   in NumPy 2.4.2. The audit note was based on an incorrect premise.
    //   Reproduction:
    //     >>> import numpy, warnings
    //     >>> warnings.simplefilter('always')
    //     >>> numpy.dtype('float')
    //     dtype('float64')
    //     # (no warnings recorded)
    //
    //   NumSharp behaviour matches NumPy — keep no test to assert deprecation.
    // -----------------------------------------------------------------------
    [TestMethod]
    public void T1_66_DtypeFloat_NotDeprecated()
    {
        // Confirms parity with NumPy 2.4.2 — np.dtype('float') resolves to
        // float64 without warning. Locks in the current behaviour so future
        // refactors do not accidentally "fix" something that is not broken.
        var d = np.dtype("float");
        d.type.Should().Be(typeof(double));
        d.typecode.Should().Be(NPTypeCode.Double);
    }

    // -----------------------------------------------------------------------
    // NEW FINDING — Unreachable `return NPTypeCode.Decimal` in ToTypeCode.
    //   File: src/NumSharp.Core/Backends/NPTypeCode.cs (line 487)
    //   Dead code after `return NPTypeCode.Complex;` — confirms Decimal has
    //   no reverse mapping. Compiler warns CS0162 but the code ships.
    //   No runtime test possible — this is a static-analysis defect that
    //   could be enforced via a roslyn analyzer if desired. Documented here
    //   for completeness.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // NEW FINDING — NPY_TYPECHAR enum value collisions (`'b'` and `'c'`).
    //   File: src/NumSharp.Core/Creation/np.dtype.cs (NPY_TYPECHAR)
    //   NPY_BYTELTR == NPY_GENBOOLLTR (both 'b'),
    //   NPY_CHARLTR == NPY_COMPLEXLTR (both 'c').
    //   Causes ambiguous ToString() and breaks switch-by-name. Surfaced by
    //   the T1.65 roundtrip tests above.
    // -----------------------------------------------------------------------
    [TestMethod, OpenBugs(IssueUrl = "audit-v2-new-typechar-collisions")]
    public void NEW_NDTypechar_EnumValueCollisions()
    {
        // The enum is internal, so we go through ToString() which surfaces the
        // collision symptom.
        var asType = typeof(NPTypeCodeExtensions);
        var toTypechar = asType.GetMethod("ToTYPECHAR", BindingFlags.NonPublic | BindingFlags.Static);

        var charLtr = toTypechar!.Invoke(null, new object[] { NPTypeCode.Char })!.ToString();
        var cmplLtr = toTypechar!.Invoke(null, new object[] { NPTypeCode.Complex })!.ToString();
        var sbyteLtr = toTypechar!.Invoke(null, new object[] { NPTypeCode.SByte })!.ToString();

        // Both 'c' map to a single underlying value — collision is observable.
        charLtr.Should().NotBe(cmplLtr,
            "NPY_CHARLTR and NPY_COMPLEXLTR share value 'c' — must split into distinct values");
        // SByte should yield NPY_BYTELTR ('b'); collision with NPY_GENBOOLLTR
        // results in unstable ToString().
        sbyteLtr.Should().NotContain("GENBOOL",
            "SByte ToTYPECHAR should resolve to NPY_BYTELTR, not NPY_GENBOOLLTR (kind code)");
    }
}
