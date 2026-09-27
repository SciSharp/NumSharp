using System;
using System.Numerics;
using NumSharp.Backends.Kernels;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     The <c>numpy.polynomial</c> additive family (plan U1): the <c>{p}domain/zero/one/x</c> constants,
    ///     <c>{p}line</c>, <c>{p}add</c>, <c>{p}sub</c>, <c>{p}trim</c> of the six bases and
    ///     <c>polyutils.as_series/trimseq/trimcoef/getdomain/mapparms/mapdomain</c>. Every expected value was
    ///     produced by NumPy 2.4.2 on CPython 3.12 (probe spelled out next to each test), including the
    ///     OBJECT-level facts the byte oracle (<c>polyseries.jsonl</c>) cannot see: which results are views,
    ///     which constants are shared, which C# argument binds which Python type.
    /// </summary>
    [TestClass]
    public class PolynomialSeriesTests
    {
        private static PowerSeriesModule P => np.polynomial.polynomial;
        private static ChebyshevModule C => np.polynomial.chebyshev;
        private static LegendreModule L => np.polynomial.legendre;
        private static LaguerreModule LA => np.polynomial.laguerre;
        private static HermiteModule H => np.polynomial.hermite;
        private static HermiteEModule HE => np.polynomial.hermite_e;
        private static PolyUtilsModule PU => np.polynomial.polyutils;

        /// <summary>Asserts dtype, shape and every byte (logical C order) against a NumPy hex dump.</summary>
        /// <param name="actual">The result.</param>
        /// <param name="dtype">NumPy's dtype name.</param>
        /// <param name="shape">NumPy's shape.</param>
        /// <param name="hex">NumPy's <c>tobytes().hex()</c>.</param>
        private static void AssertBytes(NDArray actual, string dtype, long[] shape, string hex)
        {
            actual.dtype.name.Should().Be(dtype);
            actual.shape.Should().Equal(shape);
            using var c = np.ascontiguousarray(actual);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            Convert.ToHexString(bytes).ToLowerInvariant().Should().Be(hex);
        }

        /// <summary>A double's bit pattern, for signed-zero / NaN-sign assertions.</summary>
        /// <param name="d">The value.</param>
        /// <returns>The bits.</returns>
        private static ulong Bits(double d) => (ulong)BitConverter.DoubleToInt64Bits(d);

        /// <summary>
        ///     NumPy's <c>np.nan</c> — the POSITIVE quiet NaN <c>0x7ff8000000000000</c>. .NET's <see cref="double.NaN"/>
        ///     is the negative one (<c>0xfff8…</c>), so a probe that feeds <c>np.nan</c> must be replayed with this.
        /// </summary>
        private static readonly double NpNan = BitConverter.Int64BitsToDouble(0x7ff8000000000000);

        /// <summary>NumPy's <c>np.float32(np.nan)</c> — the positive quiet NaN <c>0x7fc00000</c>.</summary>
        private static readonly float NpNanF = BitConverter.Int32BitsToSingle(0x7fc00000);

        // ---------------------------------------------------------------- constants

        [TestMethod]
        public void Constants_ValuesAndDtypes_MatchNumPy()
        {
            // polydomain = np.array([-1., 1.]); polyzero/one/x int64; lagdomain [0., 1.]; lagx [1, -1]; hermx [0, 1/2]
            AssertBytes(P.polydomain, "float64", new long[] { 2 }, "000000000000f0bf000000000000f03f");
            AssertBytes(P.polyzero, "int64", new long[] { 1 }, "0000000000000000");
            AssertBytes(C.chebone, "int64", new long[] { 1 }, "0100000000000000");
            AssertBytes(L.legx, "int64", new long[] { 2 }, "00000000000000000100000000000000");
            AssertBytes(LA.lagdomain, "float64", new long[] { 2 }, "0000000000000000000000000000f03f");
            AssertBytes(LA.lagx, "int64", new long[] { 2 }, "0100000000000000ffffffffffffffff");
            AssertBytes(H.hermx, "float64", new long[] { 2 }, "0000000000000000000000000000e03f");
            AssertBytes(HE.hermex, "int64", new long[] { 2 }, "00000000000000000100000000000000");
        }

        [TestMethod]
        public void Constants_AreSharedWriteableModuleObjects_LikeNumPy()
        {
            // NumPy: C.chebdomain is C.chebdomain -> True; writeable; a write persists (a module-level ndarray).
            ReferenceEquals(C.chebdomain, C.chebdomain).Should().BeTrue();
            ReferenceEquals(C.chebdomain, P.polydomain).Should().BeFalse("each basis module owns its own constant");
            var d = HE.hermedomain;
            d.Shape.IsWriteable.Should().BeTrue();
            try
            {
                d[0] = NDArray.Scalar(-5.0);
                HE.hermedomain.GetDouble(0).Should().Be(-5.0, "a write through the shared constant persists, as in NumPy");
            }
            finally
            {
                d[0] = NDArray.Scalar(-1.0);   // restore for every other test in the process
            }
        }

        [TestMethod]
        public void Constants_SurviveACallerDispose()
        {
            // A caller disposing the shared object must not free the buffer every later read uses.
            var z = LA.lagzero;
            z.Dispose();
            AssertBytes(LA.lagzero, "int64", new long[] { 1 }, "0000000000000000");
            ReferenceEquals(LA.lagzero, z).Should().BeTrue();
        }

        // ---------------------------------------------------------------- add / sub

        [TestMethod]
        public void AddSub_DocExamples()
        {
            // P.polyadd((1,2,3),(3,2,1)) -> [4., 4., 4.]; P.polysub((1,2,3),(3,2,1)) -> [-2., 0., 2.]
            AssertBytes(P.polyadd(new object[] { 1, 2, 3 }, new object[] { 3, 2, 1 }), "float64", new long[] { 3 },
                "000000000000104000000000000010400000000000001040");
            AssertBytes(P.polysub(new object[] { 1, 2, 3 }, new object[] { 3, 2, 1 }), "float64", new long[] { 3 },
                "00000000000000c000000000000000000000000000000040");
            // P.polysub((2,), (1,2,0)) -> [1., -2.]: the subtrahend's trailing zero is trimmed first
            AssertBytes(P.polysub(new object[] { 2 }, new object[] { 1, 2, 0 }), "float64", new long[] { 2 },
                "000000000000f03f00000000000000c0");
        }

        [TestMethod]
        public void Add_TrimmedResult_IsAView_LikeNumPy()
        {
            // C.chebadd([1,2],[1,-2]) -> array([2.]) with owndata False: trimseq SLICES the in-place result.
            var r = C.chebadd(new object[] { 1, 2 }, new object[] { 1, -2 });
            AssertBytes(r, "float64", new long[] { 1 }, "0000000000000040");
            r.Storage.IsView.Should().BeTrue();
            // An untrimmed result owns its buffer.
            P.polyadd(new object[] { 1, 2 }, new object[] { 3, 4 }).Storage.IsView.Should().BeFalse();
        }

        [TestMethod]
        public void Sub_EqualLengths_NegatesTheSubtrahendFirst()
        {
            // L.legsub(f32[1.5, nan], f64[nan, 2]): c2 = -c2 (its NaN turns NEGATIVE), then c2 += c1 —
            // bytes 000000000000f8ff | 000000000000f87f (the first slot keeps the negated NaN).
            var r = L.legsub(np.array(new[] { 1.5f, NpNanF }), np.array(new[] { NpNan, 2.0 }));
            r.dtype.name.Should().Be("float64");
            Bits(r.GetDouble(0)).Should().Be(0xfff8000000000000UL);
            Bits(r.GetDouble(1)).Should().Be(0x7ff8000000000000UL);
        }

        [TestMethod]
        public void AddSub_Dtypes_FollowCommonType()
        {
            // HE.hermeadd(c128 [1+2j, 0.5-1j], int [3,4,5]) -> complex128 [4+2j, 4.5-1j, 5+0j]
            AssertBytes(HE.hermeadd(np.array(new[] { new Complex(1, 2), new Complex(0.5, -1) }), np.array(new long[] { 3, 4, 5 })),
                "complex128", new long[] { 3 },
                "000000000000104000000000000000400000000000001240000000000000f0bf00000000000014400000000000000000");
            // LA.lagsub(f16 [0.1, 0.2], f16 [0.3]) -> float16 [-0.2001, 0.2] (the float16 loop's rounding)
            AssertBytes(LA.lagsub(np.array(new[] { (Half)0.1, (Half)0.2 }), np.array(new[] { (Half)0.3 })),
                "float16", new long[] { 2 }, "67b26632");
        }

        [TestMethod]
        public void AddSub_Errors_InNumPysOrder()
        {
            Action empty = () => P.polyadd(new double[0], new object[] { 1 });
            empty.Should().Throw<ValueError>().WithMessage("Coefficient array is empty");
            Action nd = () => P.polyadd(new int[,] { { 1 } }, new object[] { 1 });
            nd.Should().Throw<ValueError>().WithMessage("Coefficient array is not 1-d");
            Action boolean = () => P.polysub(new[] { true }, new[] { 1.0 });
            boolean.Should().Throw<ValueError>().WithMessage("Coefficient arrays have no common type");
            // the size/ndim checks of BOTH arrays run before the common type: [True] then [] reports empty
            Action order = () => P.polyadd(new[] { true }, new double[0]);
            order.Should().Throw<ValueError>().WithMessage("Coefficient array is empty");
        }

        // ---------------------------------------------------------------- trimseq / trimcoef / as_series

        [TestMethod]
        public void TrimSeq_ReturnsItselfOrAView()
        {
            var a = np.array(new[] { 1.0, double.NaN });
            ReferenceEquals(PU.trimseq(a), a).Should().BeTrue("the last element is nonzero (NaN is nonzero)");
            var b = np.array(new[] { 1.0, 0.0, -0.0 });
            var tb = PU.trimseq(b);
            tb.size.Should().Be(1);
            tb.Storage.IsView.Should().BeTrue();
            tb[0] = NDArray.Scalar(7.0);
            b.GetDouble(0).Should().Be(7.0, "the trimmed result is a view of the input");
            // np.array([0.]) -> a VIEW seq[:1] even though nothing is removed (owndata False in NumPy)
            var z = np.array(new[] { 0.0 });
            var tz = PU.trimseq(z);
            ReferenceEquals(tz, z).Should().BeFalse();
            tz.Storage.IsView.Should().BeTrue();
        }

        [TestMethod]
        public void TrimSeq_NdRows_TruthValueRules()
        {
            // trimseq(np.array([[1],[0]])) -> [[1]]; rows of 2 elements raise NumPy's truth-value error
            PU.trimseq(np.array(new long[,] { { 1 }, { 0 } })).shape.Should().Equal(new long[] { 1, 1 });
            Action two = () => PU.trimseq(np.array(new[,] { { 1.0, 2.0 }, { 0.0, 0.0 } }));
            two.Should().Throw<ValueError>().WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            Action zeroRows = () => PU.trimseq(np.zeros(new Shape(3L, 0L)));
            zeroRows.Should().Throw<ValueError>().WithMessage("The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            Action zeroD = () => PU.trimseq(NDArray.Scalar(0.0));
            zeroD.Should().Throw<TypeError>().WithMessage("len() of unsized object");
        }

        [TestMethod]
        public void TrimCoef_DocExamples()
        {
            AssertBytes(PU.trimcoef(new object[] { 0, 0, 3, 0, 5, 0, 0 }), "float64", new long[] { 5 },
                "00000000000000000000000000000000000000000000084000000000000000000000000000001440");
            AssertBytes(PU.trimcoef(new object[] { 0, 0, 1e-3, 0, 1e-5, 0, 0 }, 1e-3), "float64", new long[] { 1 }, "0000000000000000");
            // pu.trimcoef(np.array([3e-4, 1e-3*(1-1j), 5e-4, 2e-5*(1+1j)]), 2e-4) -> [0.0003+0j, 0.001-0.001j, 0.0005+0j]
            var c = np.array(new[] { new Complex(3e-4, 0), new Complex(1e-3, -1e-3), new Complex(5e-4, 0), new Complex(2e-5, 2e-5) });
            AssertBytes(PU.trimcoef(c, 2e-4), "complex128", new long[] { 3 },
                "613255302aa9333f0000000000000000fca9f1d24d62503ffca9f1d24d6250bffca9f1d24d62403f0000000000000000");
            // the {p}trim aliases are the same function
            AssertBytes(C.chebtrim(new object[] { 0, 0, 3, 0, 5, 0, 0 }), "float64", new long[] { 5 },
                "00000000000000000000000000000000000000000000084000000000000000000000000000001440");
        }

        [TestMethod]
        public void TrimCoef_NothingAboveTol_IsCoefZeroTimesZero()
        {
            // pu.trimcoef([-0.5], 1) -> [-0.]: c[:1] * 0 keeps the sign; [inf], inf -> [nan]
            Bits(PU.trimcoef(new[] { -0.5 }, 1).GetDouble(0)).Should().Be(0x8000000000000000UL);
            double.IsNaN(PU.trimcoef(new[] { double.PositiveInfinity }, double.PositiveInfinity).GetDouble(0)).Should().BeTrue();
        }

        [TestMethod]
        public void TrimCoef_ToleranceTyping_WeakVsStrong()
        {
            // float16 series: both a Python float and np.float64(1.0001) keep [1, 1.001]
            var c = np.array(new[] { (Half)1.0, (Half)1.0009765625 });
            AssertBytes(PU.trimcoef(c, 1.0001), "float16", new long[] { 2 }, "003c013c");
            AssertBytes(PU.trimcoef(c, NDArray.Scalar(1.0001)), "float16", new long[] { 2 }, "003c013c");
        }

        [TestMethod]
        public void TrimCoef_Errors()
        {
            Action neg = () => PU.trimcoef(new[] { 1.0 }, -1);
            neg.Should().Throw<ValueError>().WithMessage("tol must be non-negative");
            Action cplx = () => PU.trimcoef(new[] { 1.0 }, new Complex(1, 0));
            cplx.Should().Throw<TypeError>().WithMessage("'<' not supported between instances of 'complex' and 'int'");
            Action none = () => PU.trimcoef(new[] { 1.0 }, null);
            none.Should().Throw<TypeError>().WithMessage("'<' not supported between instances of 'NoneType' and 'int'");
            Action big = () => PU.trimcoef(new[] { 1.0, 2.0 }, BigInteger.Pow(2, 1030));
            big.Should().Throw<OverflowException>().WithMessage("int too large to convert to float");
        }

        [TestMethod]
        public void AsSeries_ConvertsAndCopies()
        {
            var r = PU.as_series(new object[] { new object[] { 1, 2 }, new object[] { 3, 4.5 } });
            r.Length.Should().Be(2);
            AssertBytes(r[0], "float64", new long[] { 2 }, "000000000000f03f0000000000000040");
            AssertBytes(r[1], "float64", new long[] { 2 }, "00000000000008400000000000001240");
            var f16 = np.array(new[] { (Half)1, (Half)0 });
            var r2 = PU.as_series(new object[] { f16, np.array(new sbyte[] { 2 }) }, trim: false);
            AssertBytes(r2[0], "float64", new long[] { 2 }, "000000000000f03f0000000000000000");
            ReferenceEquals(r2[0], f16).Should().BeFalse("as_series returns fresh copies");
            PU.as_series(np.arange(4)).Length.Should().Be(4, "a 1-D array iterates to one series per element");
            PU.as_series(new object[0]).Length.Should().Be(0);
            Action scalar = () => PU.as_series(5);
            scalar.Should().Throw<TypeError>().WithMessage("'int' object is not iterable");
            Action zeroD = () => PU.as_series(NDArray.Scalar(5.0));
            zeroD.Should().Throw<TypeError>().WithMessage("iteration over a 0-d array");
        }

        // ---------------------------------------------------------------- getdomain

        [TestMethod]
        public void GetDomain_RealComplexAndSignedZeros()
        {
            AssertBytes(PU.getdomain(new object[] { 1, 3, 2 }), "float64", new long[] { 2 }, "000000000000f03f0000000000000840");
            AssertBytes(PU.getdomain(new object[] { new Complex(1, 1), new Complex(1, -1), 0, 2 }), "complex128", new long[] { 2 },
                "0000000000000000000000000000f0bf0000000000000040000000000000f03f");
            AssertBytes(PU.getdomain(np.array(new[] { (Half)0.5, (Half)(-2.0), (Half)3.0 })), "float16", new long[] { 2 }, "00c00042");
            // [0., -0.] -> [-0., -0.] and [-0., 0.] -> [0., 0.]: NumPy's reduction lets the later zero win a tie
            AssertBytes(PU.getdomain(new[] { 0.0, -0.0 }), "float64", new long[] { 2 }, "00000000000000800000000000000080");
            AssertBytes(PU.getdomain(new[] { -0.0, 0.0 }), "float64", new long[] { 2 }, "00000000000000000000000000000000");
            Action b = () => PU.getdomain(new[] { true, false });
            b.Should().Throw<ValueError>().WithMessage("Coefficient arrays have no common type");
        }

        /// <summary>
        ///     getdomain's blocked pass (<see cref="NDPolySeries.TryDomainBlocked"/>: 8 KB windows of NumPy's copy —
        ///     read in place, or packed into an L1 scratch — folded by the exact min/max schedule) against the
        ///     copy-then-reduce route it replaced, on the same inputs, byte for byte. The copy route reduces NumPy's
        ///     C-contiguous copy with the engine's NumPy-exact schedule (oracle-proven), so this pins the one thing
        ///     the blocked pass adds: that carrying the accumulators across windows changes nothing. Every dtype the
        ///     pass serves, contiguous / stride-2 / stride-3 / reversed / reversed-stride-2 layouts, lengths across
        ///     the window boundaries of every width (a window holds 8,192 / itemsize elements and the windows start
        ///     after element 0), and data built to expose a schedule slip: sign-mixed zeros (which ±0 survives a tie
        ///     is decided lane by lane), a single NaN payload placed on the seed, on a window boundary, inside a
        ///     window and in the scalar tail (it comes back canonical or with its payload depending on where the
        ///     schedule meets it), and uint64 values that collide once converted to float64.
        /// </summary>
        [TestMethod]
        public void GetDomain_BlockedPass_MatchesTheCopyRoute_BitForBit()
        {
            var served = new[]
            {
                NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
                NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Single, NPTypeCode.Double,
            };
            // Window boundaries fall after element 1 + k·(8192 / itemsize): 1,025 / 2,049 (8-byte), 2,049 / 4,097
            // (4-byte), 4,097 / 8,193 (2-byte), 8,193 / 16,385 (1-byte).
            int[] lengths = { 1, 2, 5, 33, 1024, 1025, 1026, 2049, 2050, 4097, 8193, 8200, 16390 };
            long calls = 0, blocked = 0;
            foreach (var dt in served)
            foreach (int n in lengths)
            foreach (var (kind, values) in DomainData(dt, n))
            using (values)
            foreach (var (layout, x) in DomainLayouts(values))
            {
                calls++;
                blocked += AssertDomainRoutesAgree(x, $"{dt}/{n}/{kind}/{layout}");
            }
            blocked.Should().Be(calls, "every call of a served dtype must take the blocked pass");

            // float16 has no exact schedule: the pass declines and the copy route answers.
            using var h = Patterned(NPTypeCode.Half, 3000);
            foreach (var (layout, x) in DomainLayouts(h))
                AssertDomainRoutesAgree(x, $"Half/{layout}").Should().Be(0, "float16 is not served by the blocked pass");
        }

        /// <summary>
        ///     NumPy 2.4.2 pins for the schedule the blocked pass must carry across windows. Lane order: 2,101 float64
        ///     values of -1 with a -0.0 early in lane 3 (index 4, or 1,028 — a later window) and a +0.0 late in lane 0
        ///     (index 2,097, the last window): NumPy's max is <c>-0.0</c> — each lane keeps its own last zero and the
        ///     horizontal cascade lets the HIGHER lane win the tie — where a sequential fold would return the later
        ///     +0.0; the mirror (+1 everywhere, +0.0 early in lane 3, -0.0 late in lane 0) gives min <c>+0.0</c>. The
        ///     same bits through a stride-2 view, which NumPy copies first. NaN: one NaN with payload
        ///     <c>0x7ff8000000000bad</c> comes back canonical from a window boundary (index 1,025, inside the vector
        ///     section) but keeps its payload from the scalar tail (index 2,099). Probe:
        ///     <c>x = np.full(2101, -1.0); x[4] = -0.0; x[2097] = 0.0; pu.getdomain(x).tobytes().hex()</c> (and the
        ///     analogous calls). (Sign-mixed zeros ALONE cannot tell the schedules apart: with every element a tie, the
        ///     highest lane holds the array's last element, so NumPy and a sequential fold agree.)
        /// </summary>
        [TestMethod]
        public void GetDomain_BlockedPass_NumPyPins()
        {
            foreach (int early in new[] { 4, 1028 })
            {
                var mx = new double[2101];
                var mn = new double[2101];
                for (int i = 0; i < mx.Length; i++) { mx[i] = -1.0; mn[i] = 1.0; }
                mx[early] = -0.0; mx[2097] = 0.0;
                mn[early] = 0.0; mn[2097] = -0.0;
                using var mxc = np.array(mx);
                using var mnc = np.array(mn);
                AssertBytes(PU.getdomain(mxc), "float64", new long[] { 2 }, "000000000000f0bf0000000000000080");
                AssertBytes(PU.getdomain(mnc), "float64", new long[] { 2 }, "0000000000000000000000000000f03f");
                var mx2 = np.zeros(new Shape(4202L), NPTypeCode.Double);
                mx2["::2"] = mxc;
                AssertBytes(PU.getdomain(mx2["::2"]), "float64", new long[] { 2 }, "000000000000f0bf0000000000000080");
            }

            double payload = BitConverter.Int64BitsToDouble(0x7ff8000000000badL);
            var a = new double[2100];
            for (int i = 0; i < a.Length; i++) a[i] = (i * 7919 % 2003) - 1001;
            a[1025] = payload;
            using var nb = np.array(a);
            AssertBytes(PU.getdomain(nb), "float64", new long[] { 2 }, "000000000000f87f000000000000f87f");
            a[1025] = 0; a[2099] = payload;
            using var nt = np.array(a);
            AssertBytes(PU.getdomain(nt), "float64", new long[] { 2 }, "ad0b00000000f87fad0b00000000f87f");
        }

        /// <summary>
        ///     The blocked pass folds EVERY element exactly once: a unique maximum and a unique minimum are planted at
        ///     each position around every window edge (and the seed and the tail) of a series three windows long, for
        ///     every dtype the pass serves, contiguous / stride-2 / reversed. A window that folded too little — a group
        ///     dropped at its end, which lane-aligned data with repeating extremes never notices — or too much would
        ///     move one of them. Checked against the planted values themselves and against the copy route.
        /// </summary>
        [TestMethod]
        public void GetDomain_BlockedPass_FoldsEveryElement()
        {
            var served = new[]
            {
                NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32, NPTypeCode.UInt32,
                NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Single, NPTypeCode.Double,
            };
            foreach (var dt in served)
            {
                int itemsize = dt switch { NPTypeCode.SByte or NPTypeCode.Byte => 1, NPTypeCode.Int16 or NPTypeCode.UInt16 => 2,
                                           NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Single => 4, _ => 8 };
                int window = 8192 / itemsize;
                int n = 2 * window + 37;
                var positions = new System.Collections.Generic.SortedSet<int> { 0, 1, 2, n - 1 };
                foreach (int edge in new[] { window, 2 * window })          // windows are elements [1+k·w, (k+1)·w]
                    for (int d = -9; d <= 9; d++)
                        positions.Add(edge + d);
                for (int d = 1; d <= 9; d++)
                    positions.Add(n - 1 - d);
                // Unsigned dtypes cannot hold the negative minimum: shift the three levels up.
                double lo = dt is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64 ? 1 : -100;
                double mid = lo + 60, hi = lo + 120;
                foreach (int p in positions)
                {
                    int q = (p + n / 2) % n;   // the minimum's position: a different window
                    var v = new double[n];
                    for (int i = 0; i < n; i++) v[i] = mid;
                    v[p] = hi;
                    v[q] = lo;
                    using var values = np.array(v).astype(dt);
                    foreach (var (layout, x) in DomainLayouts(values))
                    {
                        string label = $"{dt}/p{p}/q{q}/{layout}";
                        AssertDomainRoutesAgree(x, label).Should().Be(1, label);
                        using var r = PU.getdomain(x);
                        // GetAtIndex boxes the element's own type (float32 for a float32 x); the typed getters would
                        // reinterpret its bits instead of converting them.
                        Convert.ToDouble(r.GetAtIndex(0)).Should().Be(lo, label);
                        Convert.ToDouble(r.GetAtIndex(1)).Should().Be(hi, label);
                    }
                }
            }
        }

        /// <summary>
        ///     The blocked pass honours <see cref="NumSharp.Backends.Iteration.NDExpr.DisableExactMinMax"/> like every
        ///     other exact min/max route: with it set, getdomain declines to the copy route (which then reduces with the
        ///     engine's older value-exact kernels), and the values still agree.
        /// </summary>
        [TestMethod]
        public void GetDomain_BlockedPass_DeclinesWhenExactSchedulesAreDisabled()
        {
            using var x = Patterned(NPTypeCode.Int64, 3000);
            long before = NDPolySeries.BlockedDomainRuns;
            using var exact = PU.getdomain(x["::2"]);
            (NDPolySeries.BlockedDomainRuns - before).Should().Be(1);
            NumSharp.Backends.Iteration.NDExpr.DisableExactMinMax = true;
            try
            {
                before = NDPolySeries.BlockedDomainRuns;
                using var old = PU.getdomain(x["::2"]);
                (NDPolySeries.BlockedDomainRuns - before).Should().Be(0);
                RawBytes(old).Should().Equal(RawBytes(exact));
            }
            finally
            {
                NumSharp.Backends.Iteration.NDExpr.DisableExactMinMax = false;
            }
        }

        /// <summary>
        ///     Runs getdomain through the copy route (<see cref="NDPolySeries.DisableBlockedDomain"/>) and through the
        ///     default routing, and asserts the same dtype, shape and bytes.
        /// </summary>
        /// <param name="x">The points.</param>
        /// <param name="label">Case label for failures.</param>
        /// <returns>1 when the blocked pass served the default-routed call, else 0.</returns>
        private static long AssertDomainRoutesAgree(NDArray x, string label)
        {
            NDPolySeries.DisableBlockedDomain = true;
            NDArray copy;
            try { copy = PU.getdomain(x); }
            finally { NDPolySeries.DisableBlockedDomain = false; }
            long before = NDPolySeries.BlockedDomainRuns;
            using var fast = PU.getdomain(x);
            using (copy)
            {
                fast.dtype.name.Should().Be(copy.dtype.name, label);
                fast.shape.Should().Equal(copy.shape, label);
                RawBytes(fast).Should().Equal(RawBytes(copy), label);
            }
            return NDPolySeries.BlockedDomainRuns - before;
        }

        /// <summary>
        ///     getdomain inputs of <paramref name="dt"/> and length <paramref name="n"/> built to expose a schedule
        ///     slip (see <see cref="GetDomain_BlockedPass_MatchesTheCopyRoute_BitForBit"/>).
        /// </summary>
        /// <param name="dt">The dtype.</param>
        /// <param name="n">Element count.</param>
        /// <returns>(kind, array) pairs — new C-contiguous 1-D arrays the caller disposes.</returns>
        private static System.Collections.Generic.IEnumerable<(string, NDArray)> DomainData(NPTypeCode dt, int n)
        {
            yield return ("patterned", Patterned(dt, n));
            if (dt is NPTypeCode.Single or NPTypeCode.Double)
            {
                // Sign-mixed zeros: the min/max VALUE is 0 everywhere, so only the schedule picks the sign.
                var z = new double[n];
                var rnd = new System.Random(n);
                for (int i = 0; i < n; i++) z[i] = rnd.Next(2) == 0 ? 0.0 : -0.0;
                yield return ("zeros", np.array(z).astype(dt));
                // Lane order (see GetDomain_BlockedPass_NumPyPins): a zero early in the highest lane and one of the other
                // sign late in lane 0, among values that lose to both — NumPy's cascade answers with the EARLY zero, a
                // sequential fold with the late one. For the max (negatives around) and the min (positives around).
                int lanes = 32 / (dt == NPTypeCode.Double ? 8 : 4);
                if (n > 2 * lanes + 1)
                {
                    int late = 1 + lanes * ((n - 1) / lanes - 1);   // lane 0 of the last whole vector
                    var mx = new double[n];
                    var mn = new double[n];
                    for (int i = 0; i < n; i++) { mx[i] = -1.0; mn[i] = 1.0; }
                    mx[lanes] = -0.0; mx[late] = 0.0;
                    mn[lanes] = 0.0; mn[late] = -0.0;
                    yield return ("lanes-max", np.array(mx).astype(dt));
                    yield return ("lanes-min", np.array(mn).astype(dt));
                }
                // One NaN with a payload, on the seed / a window boundary / inside a window / the tail.
                int window = 8192 / (dt == NPTypeCode.Double ? 8 : 4);
                foreach (int at in new[] { 0, 1, window, window + 1, n / 2, n - 2, n - 1 })
                {
                    if (at < 0 || at >= n)
                        continue;
                    var v = new double[n];
                    for (int i = 0; i < n; i++) v[i] = (i * 7919 % 2003) - 1001;
                    v[at] = BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000badUL));   // negative, payload
                    yield return ($"nan@{at}", np.array(v).astype(dt));
                }
            }
            if (dt == NPTypeCode.UInt64)
            {
                // Distinct uint64 values that convert to the same float64 (2^64-1 → 2^64, 2^53+1 → 2^53): the pass
                // reduces the converted copy, the copy route converts its uint64 extremes — the same doubles.
                var u = new ulong[n];
                for (int i = 0; i < n; i++)
                    u[i] = (i % 5) switch { 0 => ulong.MaxValue, 1 => ulong.MaxValue - 1, 2 => (1UL << 53) + 1, 3 => 1UL << 53, _ => (ulong)i };
                yield return ("collide", np.array(u));
            }
        }

        /// <summary>The layouts getdomain routes differently: contiguous, stride 2 and 3, reversed, reversed stride 2.</summary>
        /// <param name="values">The values (1-D, C-contiguous).</param>
        /// <returns>(name, view) pairs over the same values in logical order.</returns>
        private static (string, NDArray)[] DomainLayouts(NDArray values)
        {
            var s2 = np.zeros(new Shape(2 * values.size), values.typecode);
            s2["::2"] = values;
            var s3 = np.zeros(new Shape(3 * values.size), values.typecode);
            s3["::3"] = values;
            var r2 = np.zeros(new Shape(2 * values.size), values.typecode);
            r2["::-2"] = values;
            return new[]
            {
                ("c", values),
                ("s2", s2["::2"]),
                ("s3", s3["::3"]),
                ("r", values["::-1"].copy()["::-1"]),
                ("r2", r2["::-2"]),
            };
        }

        // ---------------------------------------------------------------- mapparms / mapdomain

        [TestMethod]
        public void MapParms_PythonDomains_AreCPythonArithmetic()
        {
            // pu.mapparms((-1,1),(-1,1)) -> (0.0, 1.0); ((1,-1),(-1,1)) -> (-0.0, -1.0)
            var (o1, s1) = PU.mapparms((-1, 1), (-1, 1));
            ((double)o1).Should().Be(0.0);
            ((double)s1).Should().Be(1.0);
            var (o2, s2) = PU.mapparms((1, -1), (-1, 1));
            Bits((double)o2).Should().Be(0x8000000000000000UL, "0 / -2 is -0.0 in CPython's int true division");
            ((double)s2).Should().Be(-1.0);
            // ((-1j,-1),(1,1j)) -> ((1+1j), (1-0j))
            var (o3, s3) = PU.mapparms((new Complex(0, -1), -1), (1, new Complex(0, 1)));
            ((Complex)o3).Should().Be(new Complex(1, 1));
            Bits(((Complex)s3).Imaginary).Should().Be(0x8000000000000000UL);
            // a zero-length Python domain is Python's ZeroDivisionError
            Action zero = () => PU.mapparms((1, 1), (0, 1));
            zero.Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            Action zeroF = () => PU.mapparms(new object[] { 1.0, 1.0 }, new object[] { 0, 1 });
            zeroF.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
        }

        [TestMethod]
        public void MapParms_ArrayDomains_AreNumPyScalarMath()
        {
            // mapparms(np.array([-1.,1.]), np.array([0, 2pi])) -> (pi, pi) as np.float64
            var (o, s) = PU.mapparms(np.array(new[] { -1.0, 1.0 }), np.array(new[] { 0.0, 2 * System.Math.PI }));
            o.Should().BeOfType<double>();
            Bits((double)o).Should().Be(0x400921fb54442d18UL);
            Bits((double)s).Should().Be(0x400921fb54442d18UL);
            // zero-length array domain: inf, never a raise
            var (oi, si) = PU.mapparms(np.array(new[] { 1.0, 1.0 }), np.array(new[] { 0.0, 1.0 }));
            double.IsNegativeInfinity((double)oi).Should().BeTrue();
            double.IsPositiveInfinity((double)si).Should().BeTrue();
            // a float32 domain yields float32 scalars (boxed as float): mapparms(f32[-1,1], (0, 2.5)) -> (1.25, 1.25)
            var (of, sf) = PU.mapparms(np.array(new[] { -1f, 1f }), (0, 2.5));
            of.Should().BeOfType<float>();
            ((float)sf).Should().Be(1.25f);
            // complex128 arrays: scalarmath's NAIVE complex product
            var (oc, sc) = PU.mapparms(np.array(new[] { new Complex(0.1, 0.7), new Complex(1.3, -0.2) }),
                                       np.array(new[] { new Complex(2.9, 0.3), new Complex(-0.4, 1.1) }));
            Bits(((Complex)oc).Real).Should().Be(0x4003dc805761_9f10UL);
            Bits(((Complex)oc).Imaginary).Should().Be(0x3ffd867c3ece2a54UL);
            Bits(((Complex)sc).Real).Should().Be(0xc000a3d70a3d70a4UL);
            Bits(((Complex)sc).Imaginary).Should().Be(0xbfec962fc962fc94UL);
        }

        [TestMethod]
        public void MapParms_TypedOverloads_MatchTheGeneralOne()
        {
            var (o, s) = PU.mapparms(new[] { -1.0, 1.0 }, new[] { 0.0, 2 * System.Math.PI });
            Bits(o).Should().Be(0x400921fb54442d18UL);
            Bits(s).Should().Be(0x400921fb54442d18UL);
            var (oi, _) = PU.mapparms(new[] { 1.0, 1.0 }, new[] { 0.0, 1.0 });
            double.IsNegativeInfinity(oi).Should().BeTrue("a double[] is a float64 array: NumPy's scalar math, not CPython's");
            var (oc, sc) = PU.mapparms(new[] { new Complex(0.1, 0.7), new Complex(1.3, -0.2) },
                                       new[] { new Complex(2.9, 0.3), new Complex(-0.4, 1.1) });
            Bits(oc.Real).Should().Be(0x4003dc8057619f10UL);
            Bits(sc.Imaginary).Should().Be(0xbfec962fc962fc94UL);
            Action shortDom = () => PU.mapparms(new[] { 0.0 }, new[] { 0.0, 1.0 });
            shortDom.Should().Throw<IndexError>().WithMessage("index 1 is out of bounds for axis 0 with size 1");
        }

        /// <summary>
        ///     A float32/float16 NumPy value meeting a Python complex is NumPy's complex64 SCALAR (NEP 50): its arithmetic
        ///     runs in float32 — scalarmath's naive product, CFLOAT_divide's reciprocal-multiply Smith, a 0-d operand's
        ///     fused product — and the value is carried exactly in NumSharp's complex128 (#569: the dtype differs, the
        ///     bits do not). Every literal is NumPy 2.4.2's (probed); the complex128 twin shows the precision matters.
        /// </summary>
        [TestMethod]
        public void Complex64Scalars_ReproduceNumPysFloat32Arithmetic()
        {
            var dom = np.array(new[] { 0.3f, -1.7f });
            var nw = (new Complex(1.1, -0.35), new Complex(-2.3, 0.9));
            void Is(Complex z, ulong re, ulong im)
            {
                Bits(z.Real).Should().Be(re);
                Bits(z.Imaginary).Should().Be(im);
            }

            // mapparms(f32 array, complex tuple): old[i] are np.float32 scalars -> complex64 off/scl
            var (o, s) = PU.mapparms(dom, nw);
            Is((Complex)o, 0x3fe2e147c0000000UL, 0xbfc4cccce0000000UL);
            Is((Complex)s, 0x3ffb333340000000UL, 0xbfe4000000000000UL);
            // a float64 domain stays complex128 — full-precision bits
            var (o2, s2) = PU.mapparms(np.array(new[] { 0.3, -1.7 }), nw);
            Is((Complex)o2, 0x3fe2e147ae147ae2UL, 0xbfc4ccccccccccccUL);
            Is((Complex)s2, 0x3ffb333333333333UL, 0xbfe4000000000000UL);

            // mapdomain at a Python complex x: complex64 scl * x is scalarmath's NAIVE product, in float32
            Is((Complex)PU.mapdomain(new Complex(0.77, -1.21), dom, nw), 0x3ff248b440000000UL, 0xc0059b22e0000000UL);
            // a float64 array x: the complex64 off/scl widen exactly into NumPy's complex128 loop
            var ra = PU.mapdomain(np.array(new[] { 0.5, -1.25, 3.0 }), dom, nw);
            ra.dtype.Should().Be(np.complex128);
            Is(ra.GetAtIndex<Complex>(0), 0x3ff70a3d80000000UL, 0xbfde666670000000UL);
            Is(ra.GetAtIndex<Complex>(1), 0xbff88f5c30000000UL, 0x3fe3ccccc8000000UL);
            Is(ra.GetAtIndex<Complex>(2), 0x4016c28f68000000UL, 0xc0004cccce000000UL);
            // a 0-d float32 x: `scl * x` is a one-element complex64 ufunc (a scalar result), `off + that` scalarmath
            var r0 = PU.mapdomain(NDArray.Scalar(1.3f), dom, nw);
            r0.ndim.Should().Be(0);
            Is(r0.GetAtIndex<Complex>(0), 0x4006666680000000UL, 0xbfef333340000000UL);

            // lagline: `off + scl` in complex64, then np.array with the Python complex -scl -> complex128
            var lg = LA.lagline(NDArray.Scalar(0.3f), new Complex(1.1, -0.35));
            lg.dtype.Should().Be(np.complex128);
            Is(lg.GetAtIndex<Complex>(0), 0x3ff6666680000000UL, 0xbfd6666660000000UL);
            Is(lg.GetAtIndex<Complex>(1), 0xbff199999999999aUL, 0x3fd6666666666666UL);
            // np.float16 scl: [complex64, float16] is NumPy's complex64 array — here complex128 with the same values
            var lr = LA.lagline(new Complex(1.1, -0.35), (Half)0.3);
            Is(lr.GetAtIndex<Complex>(0), 0x3ff66699a0000000UL, 0xbfd6666660000000UL);
            Is(lr.GetAtIndex<Complex>(1), 0xbfd3340000000000UL, 0x0000000000000000UL);
        }

        [TestMethod]
        public void MapParms_PythonComplexMeetsNpFloat64_IsCPython()
        {
            // new[1] - new[0] is a Python complex; oldlen an np.float64 0.0: `pycomplex / np.float64` is CPython's
            // complex division (complex accepts a float subclass) -> ZeroDivisionError, where NumPy's would give inf.
            Action a = () => PU.mapparms(np.array(new[] { 2.0, 2.0 }), (new Complex(0, 1), 2));
            a.Should().Throw<DivideByZeroException>().WithMessage("complex division by zero");
        }

        [TestMethod]
        public void MapParms_Errors()
        {
            Action t = () => PU.mapparms(ValueTuple.Create((object)0), (0, 1));
            t.Should().Throw<IndexError>().WithMessage("tuple index out of range");
            Action l = () => PU.mapparms(new object[] { 0 }, (0, 1));
            l.Should().Throw<IndexError>().WithMessage("list index out of range");
            Action z = () => PU.mapparms(NDArray.Scalar(5.0), (0, 1));
            z.Should().Throw<IndexError>().WithMessage("too many indices for array: array is 0-dimensional, but 1 were indexed");
            Action n = () => PU.mapparms(1.5, (0, 1));
            n.Should().Throw<TypeError>().WithMessage("'float' object is not subscriptable");
            Action s = () => PU.mapparms("ab", (0, 1));
            s.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'str' and 'str'");
            Action b = () => PU.mapparms(np.array(new[] { false, true }), (0, 1));
            b.Should().Throw<TypeError>().WithMessage("numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` operator, or the logical_xor function instead.");
            Action ov = () => PU.mapparms(np.array(new sbyte[] { 0, 1 }), (0, 300));
            ov.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
            Action ov2 = () => PU.mapparms(np.array(new long[] { 0, 1 }), (0, new BigInteger(ulong.MaxValue) + 1));
            ov2.Should().Throw<OverflowException>().WithMessage("int too big to convert");
            Action ov3 = () => PU.mapparms((0, 1), (0, BigInteger.Pow(10, 400)));
            ov3.Should().Throw<OverflowException>().WithMessage("integer division result too large for a float");
        }

        [TestMethod]
        public void MapParms_TupleFastLane_EdgesMatchNumPy()
        {
            // pu.mapparms((2**62, -2**62), (3, 5)): old[1]*new[0] leaves int64 — the lane hands over to exact ints
            var (o1, s1) = PU.mapparms((1L << 62, -(1L << 62)), (3, 5));
            Bits((double)o1).Should().Be(0x4010000000000000UL);
            Bits((double)s1).Should().Be(0xbc10000000000000UL);
            // ((0, -5), (0, 0)): both quotients are the int 0 over a negative int -> -0.0 (CPython's XOR sign)
            var (o2, s2) = PU.mapparms((0, -5), (0, 0));
            Bits((double)o2).Should().Be(0x8000000000000000UL);
            Bits((double)s2).Should().Be(0x8000000000000000UL);
            // ((2**62 + 513, 1.5), (0, 1)): the int meets a float correctly rounded (up to 2**62 + 1024), not truncated
            var (o3, s3) = PU.mapparms(((1L << 62) + 513, 1.5), (0, 1));
            Bits((double)o3).Should().Be(0x3ff0000000000000UL);
            Bits((double)s3).Should().Be(0xbc0ffffffffffffeUL);
            // ((-2**63, 2**63-1), (0.5, -0.0)): oldlen leaves int64 (2**64 - 1)
            var (o4, s4) = PU.mapparms((long.MinValue, long.MaxValue), (0.5, -0.0));
            Bits((double)o4).Should().Be(0x3fd0000000000000UL);
            Bits((double)s4).Should().Be(0xbbe0000000000000UL);
            // ((True, 0.5), (3, 5)): a Python bool is an int in CPython's arithmetic
            var (o5, s5) = PU.mapparms((true, 0.5), (3, 5));
            Bits((double)o5).Should().Be(0x401c000000000000UL);
            Bits((double)s5).Should().Be(0xc010000000000000UL);
            // ((3, 10**15 + 37), (2**53 + 1, 1)): an int dividend beyond 2**53 takes the exact long_true_divide
            var (o6, s6) = PU.mapparms((3, 1_000_000_000_000_037L), ((1L << 53) + 1, 1));
            Bits((double)o6).Should().Be(0x434000000000000eUL);
            Bits((double)s6).Should().Be(0xc02203af9ee75569UL);
            // the two ZeroDivisionError texts: int/int vs anything with a float (-0.0 is a zero divisor)
            Action mixed = () => PU.mapparms((1, 1.0), (0, 1));
            mixed.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
            Action negZero = () => PU.mapparms((0.0, -0.0), (0, 1));
            negZero.Should().Throw<DivideByZeroException>().WithMessage("float division by zero");
            Action boolInt = () => PU.mapparms((true, 1), (0, 1));
            boolInt.Should().Throw<DivideByZeroException>().WithMessage("division by zero");
            Action bigInt = () => PU.mapparms((1L << 62, 1L << 62), (1L << 62, 1));
            bigInt.Should().Throw<DivideByZeroException>().WithMessage("division by zero");
        }

        [TestMethod]
        public void MapDomain_TupleOverloads_MatchNumPy()
        {
            // pu.mapdomain(0.5, (2**62, -2**62), (3, 5)) -> 4.0 (a bailing domain, a Python float x)
            Bits((double)PU.mapdomain(0.5, (1L << 62, -(1L << 62)), (3, 5))).Should().Be(0x4010000000000000UL);
            // pu.mapdomain(-0.0, (0, -5), (0, 0)) -> 0.0: -0.0 + -0.0 * -0.0
            Bits((double)PU.mapdomain(-0.0, (0, -5), (0, 0))).Should().Be(0UL);
            // a Python complex x: CPython's complex arithmetic (the float scale joins as (f, 0.0))
            PU.mapdomain(new Complex(1, 2), (-1, 1), (0, 2)).Should().Be(new Complex(2, 2));
            var inf = (Complex)PU.mapdomain(new Complex(double.PositiveInfinity, 1), (-1, 1), (0, 2));
            Bits(inf.Real).Should().Be(0x7ff0000000000000UL);
            Bits(inf.Imaginary).Should().Be(0xfff8000000000000UL, "0 * inf is the x86 default (negative) NaN");
            // a complex domain element takes the general lane
            PU.mapdomain(new Complex(1, 2), (-1, 1), (0, new Complex(0, 1))).Should().Be(new Complex(-1, 1));
            // float32 points keep float32 against a Python domain: [4., 4.]
            AssertBytes(PU.mapdomain(np.array(new[] { 0.25f, -0.75f }), (1L << 62, -(1L << 62)), (3, 5)), "float32", new long[] { 2 },
                        "0000804000008040");
        }

        [TestMethod]
        public void MachineNumberLane_AgreesWithTheExactGeneralLane_BitForBit()
        {
            // Every route through the CPython machine-number lane — the generic tuple overloads, and the object
            // overloads fed a boxed tuple or an object[] list — must be indistinguishable from the exact general lane
            // (BigInteger-backed PolyNumbers): values bit for bit (NaN signs included), exceptions by type and text,
            // over ints around every edge the lane handles (2^31, 2^53, 2^62, the int64 limits), float specials and
            // complex values.
            var rng = new System.Random(20260927);
            long[] edges = { 0, 1, -1, 2, -5, 7, 1L << 31, -(1L << 31) - 1, (1L << 53) + 1, -(1L << 53) - 3, (1L << 62) + 513,
                             -(1L << 62), long.MaxValue, long.MinValue, long.MaxValue - 1 };
            double[] specials = { 0.0, -0.0, 0.5, -1.5, 1e308, -1e308, 5e-324, double.PositiveInfinity, double.NegativeInfinity, NpNan,
                                  BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000000UL)) };
            long L() => rng.Next(3) == 0 ? edges[rng.Next(edges.Length)] : rng.Next(3) == 0 ? rng.NextInt64() : rng.Next(-50, 50);
            double D() => rng.Next(2) == 0 ? specials[rng.Next(specials.Length)] : (rng.NextDouble() - 0.5) * System.Math.Pow(2, rng.Next(-60, 70));
            Complex Z() => new Complex(D(), rng.Next(4) == 0 ? 0.0 : D());
            var x = np.array(new[] { 0.25, -1.5, 3.0 });
            for (int i = 0; i < 400; i++)
            {
                AssertLaneAgreement(L(), L(), L(), L(), x, D(), Z());
                AssertLaneAgreement(D(), D(), D(), D(), x, D(), Z());
                AssertLaneAgreement(L(), D(), D(), L(), x, D(), Z());
                AssertLaneAgreement(Z(), L(), D(), Z(), x, D(), Z());
                AssertLaneAgreement(Z(), Z(), Z(), Z(), x, D(), Z());
                AssertLaneAgreement(rng.Next(2) == 0, L(), (ulong)rng.NextInt64() | (1UL << 63), (int)L(), x, D(), Z());
                AssertLaneAgreement((short)rng.Next(-9, 9), (float)D(), (byte)rng.Next(0, 3), (uint)rng.Next(), x, D(), Z());
            }
        }

        /// <summary>
        ///     Asserts one domain pair gives the exact general lane's mapparms / mapdomain(double) /
        ///     mapdomain(Complex) / mapdomain(NDArray) outcome through every public route: the generic tuple
        ///     overloads, and the object overloads with the domain boxed as a tuple and as an <c>object[]</c> list.
        /// </summary>
        /// <typeparam name="T0">old[0]'s type.</typeparam>
        /// <typeparam name="T1">old[1]'s type.</typeparam>
        /// <typeparam name="T2">new[0]'s type.</typeparam>
        /// <typeparam name="T3">new[1]'s type.</typeparam>
        /// <param name="a0">old[0].</param>
        /// <param name="a1">old[1].</param>
        /// <param name="b0">new[0].</param>
        /// <param name="b1">new[1].</param>
        /// <param name="xs">Array points.</param>
        /// <param name="xd">Python float point.</param>
        /// <param name="xc">Python complex point.</param>
        private static void AssertLaneAgreement<T0, T1, T2, T3>(T0 a0, T1 a1, T2 b0, T3 b1, NDArray xs, double xd, Complex xc)
        {
            string label = $"(({a0}, {a1}), ({b0}, {b1}))";
            object oldTuple = (a0, a1), newTuple = (b0, b1);
            object oldList = new object[] { a0, a1 }, newList = new object[] { b0, b1 };

            // The reference: the general lane, converted as the facades convert.
            Func<object> refParms = () =>
            {
                var (o, s) = NDPolySeries.MapParmsGeneral(oldTuple, newTuple);
                return (o.ToObject(), s.ToObject());
            };
            Func<PolyNumber, object> refDomain = px =>
            {
                var (o, s) = NDPolySeries.MapParmsGeneral(oldTuple, newTuple);
                var r = NDPolySeries.MapDomainWith(px, o, s);
                return px.Kind == PolyNumberKind.Array ? r.ToNDArray() : r.IsZeroDimArray ? PolyNumber.ScalarOf(r.Array).ToObject() : r.ToObject();
            };

            Same(() => PU.mapparms((a0, a1), (b0, b1)), refParms, "generic mapparms " + label);
            Same(() => PU.mapparms(oldTuple, newTuple), refParms, "object mapparms (tuple) " + label);
            Same(() => PU.mapparms(oldList, newList), refParms, "object mapparms (list) " + label);
            var pxd = PolyNumber.FromPython(PyScalar.Float(xd));
            Same(() => PU.mapdomain(xd, (a0, a1), (b0, b1)), () => refDomain(pxd), $"generic mapdomain({xd}) " + label);
            Same(() => PU.mapdomain(xd, oldList, newList), () => refDomain(pxd), $"object mapdomain({xd}) " + label);
            var pxc = PolyNumber.FromPython(PyScalar.Cplx(xc));
            Same(() => PU.mapdomain(xc, (a0, a1), (b0, b1)), () => refDomain(pxc), $"generic mapdomain({xc}) " + label);
            Same(() => PU.mapdomain(xs, (a0, a1), (b0, b1)), () => refDomain(PolyNumber.FromArray(xs)), "generic mapdomain(array) " + label);
            Same(() => PU.mapdomain(xs, oldList, newList), () => refDomain(PolyNumber.FromArray(xs)), "object mapdomain(array) " + label);
        }

        /// <summary>Runs a route and the reference and compares their results (bits) or exceptions (type and message).</summary>
        /// <param name="generic">The route under test.</param>
        /// <param name="general">The exact general lane.</param>
        /// <param name="label">What is being compared.</param>
        private static void Same(Func<object> generic, Func<object> general, string label)
        {
            object r1 = null, r2 = null;
            Exception e1 = null, e2 = null;
            try { r1 = generic(); } catch (Exception e) { e1 = e; }
            try { r2 = general(); } catch (Exception e) { e2 = e; }
            if (e1 != null || e2 != null)
            {
                (e1?.GetType(), e1?.Message).Should().Be((e2?.GetType(), e2?.Message), label);
                return;
            }
            Canon(r1).Should().Be(Canon(r2), label);
        }

        /// <summary>A comparable bit-level rendering of a mapparms/mapdomain result (tuples, arrays, boxed numbers).</summary>
        /// <param name="o">The result.</param>
        /// <returns>Type names and raw bits.</returns>
        private static string Canon(object o) => o switch
        {
            ValueTuple<object, object> t => "(" + Canon(t.Item1) + ", " + Canon(t.Item2) + ")",
            double d => "double:" + Bits(d).ToString("x16"),
            Complex c => "complex:" + Bits(c.Real).ToString("x16") + "," + Bits(c.Imaginary).ToString("x16"),
            NDArray a when a.typecode == NPTypeCode.Double =>
                "float64" + string.Join(",", a.shape) + ":" + string.Join(",", System.Array.ConvertAll(a.ToArray<double>(), v => Bits(v).ToString("x16"))),
            NDArray a => a.dtype.name + string.Join(",", a.shape) + ":" + a.ToString(),
            _ => o?.GetType().Name + ":" + o,
        };

        [TestMethod]
        public void MapDomain_ArraysScalarsAndBinding()
        {
            // pu.mapdomain(np.linspace(0, 2pi, 5), (0, 2pi), (-1, 1)) -> [-1., -0.5, 0., 0.5, 1.]
            AssertBytes(PU.mapdomain(np.linspace(0, 2 * System.Math.PI, 5), (0, 2 * System.Math.PI), (-1, 1)), "float64", new long[] { 5 },
                "000000000000f0bf000000000000e0bf0000000000000000000000000000e03f000000000000f03f");
            // float32 points with float32 array domains stay float32
            var f32Dom = np.array(new[] { -1f, 1f });
            AssertBytes(PU.mapdomain(np.array(new[] { 0.25f, -0.75f }), f32Dom, np.array(new[] { 0f, 3f })), "float32", new long[] { 2 }, "0000f03f0000c03e");
            // pu.mapdomain(1, f32 arrays) -> np.float32(2.5): a C# int binds the Python-number overload (weak),
            // so float32 is kept — an NDArray 0-d int32 x would have promoted to float64.
            var r = PU.mapdomain(1, f32Dom, np.array(new[] { 0f, 2.5f }));
            r.Should().BeOfType<float>();
            ((float)r).Should().Be(2.5f);
            // a 0-d float16 array: np.float16(2.25)
            AssertBytes(PU.mapdomain(NDArray.Scalar((Half)0.5), (-1, 1), (0, 3)), "float16", new long[0], "8040");
            // the typed fast path
            PU.mapdomain(0.5, new[] { -1.0, 1.0 }, new[] { 0.0, 2.0 }).Should().Be(1.5);
            // bool x: NumPy converts it with np.asanyarray -> np.float64(2.0)
            PU.mapdomain((object)true, (-1, 1), (0, 2)).Should().Be(2.0);
        }

        /// <summary>
        ///     The affine route of mapdomain (<c>NDPolySeries.MapDomainAffine</c>: the points converted block by block
        ///     into an L1 scratch, then one multiply-then-add kernel) against the fused np.evaluate pass it replaced, on
        ///     the same inputs, byte for byte — every points dtype, contiguous / stride-2 / reversed / 2-D layouts,
        ///     lengths across the 512 / 1,024 / 2,048-point block boundaries, and float64, float32 and complex128 loops
        ///     with scales that round. The fused pass is the oracle-proven reference; the run counter proves the affine
        ///     route actually served the calls it was expected to (a silent decline would pass every byte check).
        /// </summary>
        [TestMethod]
        public void MapDomain_AffineRoute_MatchesTheFusedPass_BitForBit()
        {
            var f64Old = np.array(new[] { -2.0, 3.0 });
            var f64New = np.array(new[] { 0.5, 4.0 });
            var f32Old = np.array(new[] { -2f, 3f });
            var f32New = np.array(new[] { 0.5f, 4f });
            var dtypes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Byte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char,
                NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single,
                NPTypeCode.Double,
            };
            long served = 0;
            foreach (var dt in dtypes)
            foreach (int n in new[] { 5, 37, 511, 1030, 2100 })
            {
                using var values = Patterned(dt, n);
                foreach (var (layout, x) in Layouts(values))
                {
                    // float64 loop (Python domains / float64 arrays), float32 loop (float32 arrays — float64 for the
                    // wide integers, as NumPy promotes them), complex128 loop (a Python complex domain).
                    AssertRoutesAgree(() => PU.mapdomain(x, (-1.0, 3.0), (0.5, 4.0)), $"{dt}/{n}/{layout}/pyfloat", ref served);
                    AssertRoutesAgree(() => PU.mapdomain(x, f64Old, f64New), $"{dt}/{n}/{layout}/f64", ref served);
                    AssertRoutesAgree(() => PU.mapdomain(x, f32Old, f32New), $"{dt}/{n}/{layout}/f32", ref served);
                    AssertRoutesAgree(() => PU.mapdomain(x, (-1, 3), (new Complex(0.5, 1), new Complex(4, -2))), $"{dt}/{n}/{layout}/c128", ref served);
                }
            }
            // 814 of the 1,092 calls when written; the rest are the shapes the fused pass keeps (a contiguous x on a
            // vector widen edge or of bool, the transposed 2-D layout, float16 and complex64 loops).
            served.Should().BeGreaterThan(780, "the affine route must actually serve most of these calls");
        }

        /// <summary>
        ///     The affine route multiplies and THEN adds, each rounded — NumPy's two ufunc calls — never a fused
        ///     multiply-add: <c>np.float64(-0.3) + np.float64(0.1) * np.array([3.0])</c> is <c>5.551115123125783e-17</c>
        ///     where <c>fma(0.1, 3, -0.3)</c> is <c>2.7755575615628914e-17</c>. The same over random points, at float64
        ///     and float32, with the check that the chosen inputs do separate the two spellings.
        /// </summary>
        [TestMethod]
        public void MapDomain_AffineRoute_RoundsMultiplyAndAddSeparately()
        {
            // NumPy 2.4.2: (np.float64(-0.3) + np.float64(0.1) * np.array([3.0])).tobytes().hex()
            using var three = np.array(new[] { 3.0 });
            var r = NDPolySeries.MapDomainWith(PolyNumber.FromArray(three), PolyNumber.FromScalar(-0.3), PolyNumber.FromScalar(0.1)).Array;
            Bits(r.GetDouble(0)).Should().Be(Bits(5.551115123125783e-17));

            var rnd = new System.Random(20260928);
            var xs = new double[4099];
            var xf = new float[4099];
            for (int i = 0; i < xs.Length; i++) { xs[i] = rnd.NextDouble() * 200 - 100; xf[i] = (float)xs[i]; }
            double scl = 0.1, off = -0.3;
            float sclF = 0.1f, offF = -0.3f;
            using var x64 = np.array(xs);
            using var x32 = np.array(xf);
            long before = NDPolySeries.AffineMapDomainRuns;
            using var r64 = NDPolySeries.MapDomainWith(PolyNumber.FromArray(x64), PolyNumber.FromScalar(off), PolyNumber.FromScalar(scl)).Array;
            using var r32 = NDPolySeries.MapDomainWith(PolyNumber.FromArray(x32), PolyNumber.FromScalar(offF), PolyNumber.FromScalar(sclF)).Array;
            (NDPolySeries.AffineMapDomainRuns - before).Should().Be(2, "both calls take the affine route");
            int fmaDiffers64 = 0, fmaDiffers32 = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                double want = off + scl * xs[i];
                Bits(r64.GetDouble(i)).Should().Be(Bits(want), $"float64 point {i}");
                if (Bits(System.Math.FusedMultiplyAdd(scl, xs[i], off)) != Bits(want)) fmaDiffers64++;
                float wantF = offF + sclF * xf[i];
                BitConverter.SingleToInt32Bits(r32.GetSingle(i)).Should().Be(BitConverter.SingleToInt32Bits(wantF), $"float32 point {i}");
                if (BitConverter.SingleToInt32Bits(MathF.FusedMultiplyAdd(sclF, xf[i], offF)) != BitConverter.SingleToInt32Bits(wantF)) fmaDiffers32++;
            }
            fmaDiffers64.Should().BeGreaterThan(0, "the inputs must distinguish a fused multiply-add at float64");
            fmaDiffers32.Should().BeGreaterThan(0, "the inputs must distinguish a fused multiply-add at float32");
        }

        /// <summary>
        ///     Runs one mapdomain call through the fused pass and through the affine route and asserts the same dtype,
        ///     shape and bytes; counts the calls the affine route served.
        /// </summary>
        /// <param name="call">The mapdomain call.</param>
        /// <param name="label">Case label for failures.</param>
        /// <param name="served">Incremented when the affine route served the second run.</param>
        private static void AssertRoutesAgree(Func<object> call, string label, ref long served)
        {
            NDPolySeries.DisableAffineMapDomain = true;
            NDArray fused;
            try { fused = (NDArray)call(); }
            finally { NDPolySeries.DisableAffineMapDomain = false; }
            long before = NDPolySeries.AffineMapDomainRuns;
            using var affine = (NDArray)call();
            served += NDPolySeries.AffineMapDomainRuns - before;
            using (fused)
            {
                affine.dtype.name.Should().Be(fused.dtype.name, label);
                affine.shape.Should().Equal(fused.shape, label);
                RawBytes(affine).Should().Equal(RawBytes(fused), label);
            }
        }

        /// <summary>The bytes of <paramref name="a"/> in logical C order.</summary>
        /// <param name="a">The array.</param>
        /// <returns>Its bytes.</returns>
        private static byte[] RawBytes(NDArray a)
        {
            using var c = np.ascontiguousarray(a);
            var bytes = new byte[c.size * c.dtypesize];
            unsafe
            {
                var p = (byte*)c.Storage.Address + c.Shape.offset * c.dtypesize;
                for (int i = 0; i < bytes.Length; i++) bytes[i] = p[i];
            }
            return bytes;
        }

        /// <summary>
        ///     <paramref name="n"/> values of <paramref name="dt"/> covering its range: integers from both extremes
        ///     through zero, floats with ±0, ±inf, NaN, subnormals and large magnitudes among ordinary values.
        /// </summary>
        /// <param name="dt">The dtype.</param>
        /// <param name="n">Element count.</param>
        /// <returns>A new C-contiguous 1-D array.</returns>
        private static NDArray Patterned(NPTypeCode dt, int n)
        {
            var d = new double[n];
            for (int i = 0; i < n; i++)
                d[i] = (i % 13) switch
                {
                    0 => 0.0,
                    1 => -0.0,
                    2 => double.PositiveInfinity,
                    3 => double.NegativeInfinity,
                    4 => double.NaN,
                    5 => 5e-324,
                    6 => 1.5e300,
                    7 => -65504.0,
                    _ => (i * 7919 % 2003 - 1001) * 0.37,
                };
            if (dt is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double)
                return np.array(d).astype(dt);
            // Integers: the extremes of the dtype and ordinary values (the float specials would wrap arbitrarily).
            var v = new long[n];
            for (int i = 0; i < n; i++)
                v[i] = (i % 7) switch { 0 => long.MinValue, 1 => long.MaxValue, 2 => 0, 3 => -1, _ => i * 7919L % 20011 - 10005 };
            return np.array(v).astype(dt);
        }

        /// <summary>The layouts the affine route serves (1-D any stride, C-contiguous N-D) plus one it must decline.</summary>
        /// <param name="values">The values (1-D, C-contiguous).</param>
        /// <returns>(name, view) pairs over the same values.</returns>
        private static (string, NDArray)[] Layouts(NDArray values)
        {
            var s2 = np.zeros(new Shape(2 * values.size), values.typecode);
            s2["::2"] = values;
            var list = new System.Collections.Generic.List<(string, NDArray)>
            {
                ("c", values),
                ("s2", s2["::2"]),
                ("r", values["::-1"].copy()["::-1"]),
            };
            if (values.size % 5 == 0)
            {
                list.Add(("2d", values.reshape(5, values.size / 5)));
                list.Add(("2dT", values.reshape(5, values.size / 5).T));   // non-contiguous N-D: the fused pass
            }
            return list.ToArray();
        }

        // ---------------------------------------------------------------- {p}line

        [TestMethod]
        public void Line_DocExamples_AllBases()
        {
            AssertBytes(P.polyline(1, -1), "int64", new long[] { 2 }, "0100000000000000ffffffffffffffff");
            AssertBytes(C.chebline(3, 2), "int64", new long[] { 2 }, "03000000000000000200000000000000");
            AssertBytes(L.legline(3, 2), "int64", new long[] { 2 }, "03000000000000000200000000000000");
            AssertBytes(LA.lagline(0, 1), "int64", new long[] { 2 }, "0100000000000000ffffffffffffffff");
            AssertBytes(H.hermline(3, 2), "float64", new long[] { 2 }, "0000000000000840000000000000f03f");
            AssertBytes(HE.hermeline(3, 2), "int64", new long[] { 2 }, "03000000000000000200000000000000");
            AssertBytes(P.polyline(1, 0.0), "int64", new long[] { 1 }, "0100000000000000");
            AssertBytes(P.polyline(1, NpNan), "float64", new long[] { 2 }, "000000000000f03f000000000000f87f");
        }

        [TestMethod]
        public void Line_ScalarKinds_DiscoverNumPysDtype()
        {
            // hermline(np.float16(1), np.float16(3)) -> float16 [1, 1.5]; a C# Half is a NumPy float16 scalar
            AssertBytes(H.hermline((Half)1, (Half)3), "float16", new long[] { 2 }, "003c003e");
            // hermline(np.float16(1), 2.0) -> float64 (the Python float 1.0 is discovered as float64)
            AssertBytes(H.hermline((Half)1, 2.0), "float64", new long[] { 2 }, "000000000000f03f000000000000f03f");
            // hermline(1, 1+2j) / (1, complex(inf, 1)): CPython 3.12's _Py_c_quot by the int 2
            AssertBytes(H.hermline(1, new Complex(1, 2)), "complex128", new long[] { 2 },
                "000000000000f03f0000000000000000000000000000e03f000000000000f03f");
            AssertBytes(H.hermline(1, new Complex(double.PositiveInfinity, 1)), "complex128", new long[] { 2 },
                "000000000000f03f0000000000000000000000000000f07f000000000000f8ff");
            // hermline(1, 2**60+1): Python's correctly rounded int/int
            AssertBytes(H.hermline(1, (1L << 60) + 1), "float64", new long[] { 2 }, "000000000000f03f000000000000a043");
            // lagline(np.int8(100), np.int8(100)) wraps in int8
            AssertBytes(LA.lagline(NDArray.Scalar((sbyte)100), NDArray.Scalar((sbyte)100)), "int8", new long[] { 2 }, "c89c");
            // polyline(-1, 2**63) -> float64 (int64 + uint64 discovery); polyline(2**63, 2**63) -> uint64
            AssertBytes(P.polyline(-1, 9223372036854775808UL), "float64", new long[] { 2 }, "000000000000f0bf000000000000e043");
            AssertBytes(P.polyline(9223372036854775808UL, 9223372036854775808UL), "uint64", new long[] { 2 },
                "00000000000000800000000000000080");
        }

        [TestMethod]
        public void Line_Errors()
        {
            Action ov = () => LA.lagline(NDArray.Scalar((sbyte)1), 300);
            ov.Should().Throw<OverflowException>().WithMessage("Python integer 300 out of bounds for int8");
            Action neg = () => LA.lagline(NDArray.Scalar(true), NDArray.Scalar(true));
            neg.Should().Throw<TypeError>().WithMessage("The numpy boolean negative, the `-` operator, is not supported, use the `~` operator or the logical_not function instead.");
            Action arr = () => P.polyline(1, np.array(new long[] { 1, 2 }));
            arr.Should().Throw<ValueError>().WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            Action inh = () => P.polyline(1, np.array(new long[] { 2 }));
            inh.Should().Throw<ValueError>().WithMessage("setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.");
            Action big = () => H.hermline(1, BigInteger.Pow(2, 1030));
            big.Should().Throw<OverflowException>().WithMessage("integer division result too large for a float");
            Action obj = () => P.polyline(BigInteger.Pow(2, 70), 1);
            obj.Should().Throw<NotSupportedException>("NumPy builds an object array, a dtype NumSharp does not have");
        }

        // ---------------------------------------------------------------- CPython arithmetic (PyScalar)

        [TestMethod]
        public void PyScalar_IntTrueDivide_IsCPythonsCorrectlyRoundedQuotient()
        {
            // (a / b).hex() from CPython 3.12
            Bits(PyScalar.IntTrueDivide(BigInteger.Pow(2, 70) + 1, 3)).Should().Be(Bits(System.Math.ScaleB(0x15555555555555, 68 - 52)));
            Bits(PyScalar.IntTrueDivide(BigInteger.Pow(10, 30), 7)).Should().Be(Bits(System.Math.ScaleB(0x1cd98a8b00a10b, 96 - 52)));
            Bits(PyScalar.IntTrueDivide(-BigInteger.Pow(2, 64) - 1, (1L << 53) + 1)).Should().Be(Bits(-System.Math.ScaleB(0x1fffffffffffff, 10 - 52)));
            Bits(PyScalar.IntTrueDivide(1, BigInteger.Pow(10, 320))).Should().Be(Bits(System.Math.ScaleB(0x7e8, -1022 - 52)));
            Bits(PyScalar.IntTrueDivide(BigInteger.Pow(2, 1100), BigInteger.Pow(2, 80))).Should().Be(Bits(System.Math.ScaleB(1.0, 1020)));
            Bits(PyScalar.IntTrueDivide(0, -5)).Should().Be(0x8000000000000000UL, "0 / -5 is -0.0");
            Action z = () => PyScalar.IntTrueDivide(1, 0);
            z.Should().Throw<DivideByZeroException>().WithMessage("division by zero");
        }

        [TestMethod]
        public void PyScalar_NaNOperandPriority_IsCPythons()
        {
            // CPython 3.12 (MSVC x64), probed with distinguishable NaNs: float_add keeps the RIGHT operand's NaN,
            // float_sub / float_div / a specialized float_mul the LEFT one's, the complex C helpers the left one's.
            double pos = NpNan, neg = BitConverter.Int64BitsToDouble(unchecked((long)0xfff8000000000000UL));
            double pay = BitConverter.Int64BitsToDouble(0x7ff8000000000001L);
            Bits(PyScalar.FloatAdd(neg, pos)).Should().Be(0x7ff8000000000000UL);
            Bits(PyScalar.FloatAdd(pos, neg)).Should().Be(0xfff8000000000000UL);
            Bits(PyScalar.FloatAdd(1.0, neg)).Should().Be(0xfff8000000000000UL);
            Bits(PyScalar.FloatSub(neg, pos)).Should().Be(0xfff8000000000000UL);
            Bits(PyScalar.FloatSub(pay, neg)).Should().Be(0x7ff8000000000001UL);
            Bits(PyScalar.FloatMul(pay, neg)).Should().Be(0x7ff8000000000001UL);
            Bits(PyScalar.FloatMul(neg, pay)).Should().Be(0xfff8000000000000UL);
            Bits(PyScalar.FloatDiv(neg, pay)).Should().Be(0xfff8000000000000UL);
            // complex(neg, 1) + complex(pos, 2) -> real neg; complex(neg, 1) * complex(pos, 2) -> (neg, neg)
            var sum = PyScalar.Apply(BinaryOp.Add, PyScalar.Cplx(new Complex(neg, 1)), PyScalar.Cplx(new Complex(pos, 2))).C;
            Bits(sum.Real).Should().Be(0xfff8000000000000UL);
            var prod = PyScalar.Apply(BinaryOp.Multiply, PyScalar.Cplx(new Complex(neg, 1)), PyScalar.Cplx(new Complex(pos, 2))).C;
            Bits(prod.Real).Should().Be(0xfff8000000000000UL);
            Bits(prod.Imaginary).Should().Be(0xfff8000000000000UL);
            // (1+2j) / complex(nan, 1): _Py_c_quot's NaN branch is Py_NAN, the POSITIVE quiet NaN
            var q = PyScalar.ComplexQuotient(new Complex(1, 2), new Complex(NpNan, 1));
            Bits(q.Real).Should().Be(0x7ff8000000000000UL);
            Bits(q.Imaginary).Should().Be(0x7ff8000000000000UL);
            // found by the tuple-overload property test: off is the default (negative) NaN of -inf/inf, scl*x the
            // positive NaN x — float_add keeps x's. pu.mapdomain(nan, (0.5, inf), (-4.983958996782492e+17,
            // 0.22046047936680752)) -> 0x7ff8000000000000 on both lanes.
            Bits((double)PU.mapdomain(NpNan, (0.5, double.PositiveInfinity), (-4.983958996782492E+17, 0.22046047936680752)))
                .Should().Be(0x7ff8000000000000UL);
            Bits((double)PU.mapdomain(NpNan, (object)(0.5, double.PositiveInfinity), (object)(-4.983958996782492E+17, 0.22046047936680752)))
                .Should().Be(0x7ff8000000000000UL);
        }

        [TestMethod]
        public void PyScalar_ComplexQuotient_IsCPython312()
        {
            var q = PyScalar.ComplexQuotient(new Complex(1, 2), new Complex(3, -4));
            Bits(q.Real).Should().Be(Bits(-System.Math.ScaleB(0x1999999999999a, -3 - 52)));
            Bits(q.Imaginary).Should().Be(Bits(System.Math.ScaleB(0x1999999999999a, -2 - 52)));
            var inf = PyScalar.ComplexQuotient(new Complex(1e300, 1e300), new Complex(1e-300, 2e-300));
            double.IsPositiveInfinity(inf.Real).Should().BeTrue();
            double.IsNegativeInfinity(inf.Imaginary).Should().BeTrue();
            // no Annex G recovery in 3.12: (1+1j)/(inf+infj) is (nan+nanj)
            var nan = PyScalar.ComplexQuotient(new Complex(1, 1), new Complex(double.PositiveInfinity, double.PositiveInfinity));
            double.IsNaN(nan.Real).Should().BeTrue();
            Action z = () => PyScalar.ComplexQuotient(1, Complex.Zero);
            z.Should().Throw<DivideByZeroException>().WithMessage("complex division by zero");
        }
    }
}
