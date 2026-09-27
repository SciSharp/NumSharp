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
