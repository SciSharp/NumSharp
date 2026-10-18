using System;
using System.Numerics;
using NumSharp.Backends.Kernels;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Polynomial
{
    /// <summary>
    ///     <c>numpy.polynomial.polynomial.polyvalfromroots</c> (plan <c>docs/plans/numpy-polynomial-remaining-compute.md</c>
    ///     §3.1, unit 6): NumPy 2.4.2's <c>np.prod(x - r, axis=0)</c> after its prologue, computed by U3's root-product kernel.
    ///     Every expected value is NumPy's (the probe scripts <c>pvfr_probe1.py</c>, <c>pvfr_probe2.py</c> and
    ///     <c>pvfr_pins.py</c> printed them); the oracle tier <c>polyeval.jsonl</c> (section M) holds the full matrix.
    /// </summary>
    /// <remarks>
    ///     The complex and float16 pins are chosen so NumPy's two multiply loops give different bits: which one NumPy runs
    ///     depends on the call's SHAPE (its reduction iterator), so the same roots and point give different results through
    ///     <c>polyvalfromroots(z, r)</c> and <c>polyvalfromroots([z, w], r)[0]</c>.
    /// </remarks>
    [TestClass]
    public class PolynomialValFromRootsTests
    {
        /// <summary>The facade under test (one call site, so a signature change shows here only).</summary>
        /// <param name="x">The points, any argument kind the facade takes.</param>
        /// <param name="r">The roots, any argument kind the facade takes.</param>
        /// <param name="tensor">NumPy's tensor flag.</param>
        /// <returns>The facade's result.</returns>
        private static NDArray Vfr(object x, object r, bool tensor = true) => np.polynomial.polynomial.polyvalfromroots(x, r, tensor);

        /// <summary>A complex from the raw bits of its parts (the pins are bit patterns, so no decimal rounding enters).</summary>
        /// <param name="re">The real part's IEEE bits.</param>
        /// <param name="im">The imaginary part's IEEE bits.</param>
        /// <returns>The complex.</returns>
        private static Complex C(ulong re, ulong im)
            => new Complex(BitConverter.Int64BitsToDouble((long)re), BitConverter.Int64BitsToDouble((long)im));

        /// <summary>The raw bits of a complex's parts — what the pins compare (a -0.0 or a NaN payload counts).</summary>
        /// <param name="z">The value.</param>
        /// <returns>(real bits, imaginary bits).</returns>
        private static (ulong, ulong) Bits(Complex z)
            => ((ulong)BitConverter.DoubleToInt64Bits(z.Real), (ulong)BitConverter.DoubleToInt64Bits(z.Imaginary));

        /// <summary>The bytes of an array in C order, as lowercase hex (NumPy's <c>tobytes().hex()</c>).</summary>
        /// <param name="a">The array (any layout).</param>
        /// <returns>The hex string.</returns>
        private static string Hex(NDArray a) => Convert.ToHexString(np.ascontiguousarray(a).tobytes()).ToLowerInvariant();

        /// <summary>A float16 from its bit pattern.</summary>
        /// <param name="bits">The IEEE half-precision bits.</param>
        /// <returns>The value.</returns>
        private static Half H(ushort bits) => BitConverter.UInt16BitsToHalf(bits);

        // pvfr_pins.py: np.random.default_rng(1); rs, rs2 = uniform(-2, 2, 5) + 1j*uniform(-2, 2, 5) each; z one draw.
        /// <summary>Five full-mantissa complex roots whose two NumPy product forms differ at <see cref="Z"/>.</summary>
        private static readonly Complex[] Roots =
        {
            C(0x3fa835ef9bc91700UL, 0xbfd3a0e0ba218790UL), C(0x3ffcd465aef054aaUL, 0x3ff4f9144cd17b9aUL),
            C(0xbff6c616c27dc4faUL, 0xbfd73eb9b3be35c8UL), C(0x3ffcb6ac2bb4ab1aUL, 0x3fc964580549a3f0UL),
            C(0xbfe815e82fe01948UL, 0xbffe3c78b37a8340UL),
        };

        /// <summary>A second such set (the other column of the tensor=False cases).</summary>
        private static readonly Complex[] Roots2 =
        {
            C(0x3ff0398f0bb765ccUL, 0xbfc7cf1fe7c04bf0UL), C(0x3fc3878535acff30UL, 0xbff76bdc5f817a60UL),
            C(0xbfe5cb59e2975ab4UL, 0xbfd8cd965a9ddf50UL), C(0x3ff2759daa1cf8b4UL, 0xbff2fa96dec626f8UL),
            C(0xbfe930e96df02374UL, 0xbfee6c8436bfb604UL),
        };

        /// <summary>The point the complex pins are evaluated at.</summary>
        private static readonly Complex Z = C(0x3ff005f98c4144c4UL, 0xbfec1b90d9780fb8UL);

        /// <summary>
        ///     complex128: one result element (a Python-scalar, 0-d or one-point x) is NumPy's reduce loop — CDOUBLE_multiply's
        ///     MSVC-contracted fallback product; several points are its elementwise loop — simd_cmul. The two differ in the last
        ///     bit of the real part here (…2b vs …2d).
        /// </summary>
        [TestMethod]
        public void Complex_ProductForm_FollowsNumPysIteration()
        {
            var r = np.array(Roots);
            (ulong, ulong) reduce = (0xc0187a6faf12402bUL, 0x40304adc204ff61aUL);
            (ulong, ulong) slab = (0xc0187a6faf12402dUL, 0x40304adc204ff61aUL);

            Bits((Complex)Vfr(Z, r)).Should().Be(reduce);
            Bits((Complex)Vfr(np.array(Z), r)).Should().Be(reduce);
            Bits(Vfr(np.array(new[] { Z }), r).GetComplex(0)).Should().Be(reduce);

            var two = Vfr(np.array(new[] { Z, new Complex(0.5, 0.5) }), r);
            Bits(two.GetComplex(0)).Should().Be(slab);
            Bits(two.GetComplex(1)).Should().Be((0xc024d1721b99a35bUL, 0x4004c07c1adc8893UL));
        }

        /// <summary>
        ///     tensor=False: an F-ordered r makes the roots axis innermost in NumPy's <c>x - r</c> temporary, so np.prod's
        ///     inner loop walks the roots (the reduce product) for EVERY output; the same r C-ordered walks the outputs.
        /// </summary>
        [TestMethod]
        public void Complex_TensorFalse_RootOrderPicksTheForm()
        {
            var stacked = np.stack(new[] { np.array(Roots), np.array(Roots2) }, axis: 1);   // (5, 2)
            var rF = np.asfortranarray(stacked);
            var rC = np.ascontiguousarray(stacked);
            var xs = np.array(new[] { Z, Z });

            var f = Vfr(xs, rF, tensor: false);
            Bits(f.GetComplex(0)).Should().Be((0xc0187a6faf12402bUL, 0x40304adc204ff61aUL));
            Bits(f.GetComplex(1)).Should().Be((0x3fe1204126242c24UL, 0x3fe1b022ce02e894UL));
            var c = Vfr(xs, rC, tensor: false);
            Bits(c.GetComplex(0)).Should().Be((0xc0187a6faf12402dUL, 0x40304adc204ff61aUL));
            Bits(c.GetComplex(1)).Should().Be((0x3fe1204126242c24UL, 0x3fe1b022ce02e895UL));

            // tensor=True at ONE point: the temporary is (5, 2, 1), and its axis 1 (the columns) decides — the C r's columns
            // are innermost (slab), the F r's roots (reduce), though the result has two elements either way.
            var one = np.array(new[] { Z });
            var tc = Vfr(one, rC);
            tc.Should().BeShaped(2, 1);
            Bits(tc.GetComplex(0, 0)).Should().Be((0xc0187a6faf12402dUL, 0x40304adc204ff61aUL));
            Bits(tc.GetComplex(1, 0)).Should().Be((0x3fe1204126242c24UL, 0x3fe1b022ce02e895UL));
            var tf = Vfr(one, rF);
            Bits(tf.GetComplex(0, 0)).Should().Be((0xc0187a6faf12402bUL, 0x40304adc204ff61aUL));
            Bits(tf.GetComplex(1, 0)).Should().Be((0x3fe1204126242c24UL, 0x3fe1b022ce02e894UL));
            // A stepped C-ordered r is no contiguous array, so the layout vote decides: its columns are still innermost (slab),
            // at one point and at a scalar x alike.
            var stepped = np.stack(new[] { np.array(Roots), np.array(Roots2), np.array(Roots), np.array(Roots2) }, axis: 1)[":, ::2"];
            foreach (var y in new[] { Vfr(one, stepped).reshape(2), Vfr(Z, stepped) })
            {
                Bits(y.GetComplex(0)).Should().Be((0xc0187a6faf12402dUL, 0x40304adc204ff61aUL));
                Bits(y.GetComplex(1)).Should().Be((0xc0187a6faf12402dUL, 0x40304adc204ff61aUL));
            }
        }

        /// <summary>
        ///     float16: the reduce loop (HALF_multiply's reduce branch) accumulates every factor in float32 and narrows ONCE;
        ///     the elementwise loop narrows after every root. Same roots, same point: 0x2417 against 0x2418.
        /// </summary>
        [TestMethod]
        public void Float16_ProductForm_FollowsNumPysIteration()
        {
            var r = np.array(new[] { H(0x34cd), H(0xb9ae), H(0x3c85), H(0xbcae), H(0x388f), H(0x3b5a), H(0xb554) });
            var x = H(0x3a29);   // 0.77

            BitConverter.HalfToUInt16Bits(Vfr(np.array(new[] { x }), r).GetHalf(0)).Should().Be(9239);
            BitConverter.HalfToUInt16Bits((Half)Vfr(0.77, r)).Should().Be(9239);
            var two = Vfr(np.array(new[] { x, x }), r);
            BitConverter.HalfToUInt16Bits(two.GetHalf(0)).Should().Be(9240);
            BitConverter.HalfToUInt16Bits(two.GetHalf(1)).Should().Be(9240);

            // tensor=False, the roots and their reverse as two columns: F (reduce) against C (slab).
            var reversed = np.array(new[] { H(0xb554), H(0x3b5a), H(0x388f), H(0xbcae), H(0x3c85), H(0xb9ae), H(0x34cd) });
            var stacked = np.stack(new[] { r, reversed }, axis: 1);
            var f = Vfr(np.array(new[] { x, x }), np.asfortranarray(stacked), tensor: false);
            BitConverter.HalfToUInt16Bits(f.GetHalf(0)).Should().Be(9239);
            BitConverter.HalfToUInt16Bits(f.GetHalf(1)).Should().Be(9239);
            var c = Vfr(np.array(new[] { x, x }), np.ascontiguousarray(stacked), tensor: false);
            BitConverter.HalfToUInt16Bits(c.GetHalf(0)).Should().Be(9240);
            BitConverter.HalfToUInt16Bits(c.GetHalf(1)).Should().Be(9238);
        }

        /// <summary>NumPy's docstring examples, and a 1-D r at an array x.</summary>
        [TestMethod]
        public void DocstringExamples()
        {
            var r = np.arange(-2, 2).reshape(2, 2);
            var t = Vfr(new object[] { -2L, 1L }, r);
            t.Should().BeShaped(2, 2);
            t.ToArray<double>().Should().Equal(-0.0, 3.0, 3.0, 0.0);
            double.IsNegative(t.GetDouble(0, 0)).Should().BeTrue();   // (-2 - -2) * (-2 - 0) is -0.0
            var b = Vfr(new object[] { -2L, 1L }, r, tensor: false);
            b.ToArray<double>().Should().Equal(-0.0, 0.0);
            Vfr(np.arange(5.0), new object[] { 1.0, 2.0 }).ToArray<double>().Should().Equal(2.0, -0.0, 0.0, 2.0, 6.0);
        }

        /// <summary>
        ///     The result's layout is the reduction output of NumPy's <c>x - r</c> temporary: x's memory order under tensor
        ///     (an F x gives an F result), r's column order for a scalar x.
        /// </summary>
        [TestMethod]
        public void ResultLayout_IsNumPys()
        {
            var xf = np.asfortranarray(np.arange(6.0).reshape(2, 3));
            var a = Vfr(xf, new object[] { 1.0, 2.0 });
            a.Shape.IsFContiguous.Should().BeTrue();
            a.Shape.strides.Should().Equal(1L, 2L);

            var r3 = np.arange(24.0).reshape(2, 3, 4).transpose(new[] { 0, 2, 1 });   // (2, 4, 3), strides (12, 1, 4)
            var b = Vfr(np.arange(5.0), r3);
            b.Should().BeShaped(4, 3, 5);
            b.Shape.strides.Should().Equal(5L, 20L, 1L);
            var c = Vfr(2.0, r3);
            c.Shape.strides.Should().Equal(1L, 4L);
            // NumPy's c.ravel() (C order, whatever the strides).
            c.ToArray<double>().Should().Equal(-20.0, 28.0, 108.0, -11.0, 45.0, 133.0, -0.0, 64.0, 160.0, 13.0, 85.0, 189.0);
        }

        /// <summary>
        ///     An integer r reaches x - r as <c>r.astype(np.double)</c>, a copy in the K order — which for a broadcast r sorts
        ///     the stride-0 axis innermost, where the float64 view itself would abstain. So the same broadcast roots give an
        ///     F-like result as int64 and a C one as float64.
        /// </summary>
        [TestMethod]
        public void IntegerRoots_AreLaidOutByAstypesKOrder()
        {
            var bi = np.arange(12).astype(NPTypeCode.Int64).reshape(4, 1, 3);
            var ri = np.broadcast_to(bi, new Shape(4, 2, 3));
            var rf = np.broadcast_to(bi.astype(NPTypeCode.Double), new Shape(4, 2, 3));

            Vfr(0.75, ri).Shape.strides.Should().Equal(1L, 2L);
            Vfr(0.75, rf).Shape.strides.Should().Equal(3L, 1L);
            Vfr(np.arange(3.0), ri).Shape.strides.Should().Equal(3L, 6L, 1L);
            Vfr(np.arange(3.0), rf).Shape.strides.Should().Equal(9L, 3L, 1L);
            Vfr(0.75, ri).ToArray<double>().Should().Equal(Vfr(0.75, rf).ToArray<double>());
        }

        /// <summary>NEP 50 dtypes: a Python-scalar x adopts the roots' dtype, a 0-d array is strong, integer roots are float64.</summary>
        [TestMethod]
        public void Dtypes_FollowNep50()
        {
            var f32 = Vfr(2.0, np.array(new float[] { 1.5f, 0.1f }));
            f32.typecode.Should().Be(NPTypeCode.Single);
            Hex(f32).Should().Be("3333733f");
            Vfr(true, np.array(new[] { H(0x3c00) })).typecode.Should().Be(NPTypeCode.Half);
            Vfr(np.array((short)2), np.array(new[] { H(0x3c00), H(0x4000) })).typecode.Should().Be(NPTypeCode.Single);
            Vfr(np.array(new sbyte[] { 1, 2 }), np.array(new[] { H(0x3c00) })).typecode.Should().Be(NPTypeCode.Half);
            Vfr(2L, new object[] { 1L, 2L }).typecode.Should().Be(NPTypeCode.Double);
            // A Python int adopts a float16 loop: 10**6 is inf there (NumPy only warns).
            Half.IsPositiveInfinity((Half)Vfr(1000000L, np.array(new[] { H(0x3c00) }))).Should().BeTrue();
            // A Python int past uint64 is exact CPython arithmetic before its float conversion.
            Vfr(BigInteger.Pow(2, 70), new object[] { 1.0 }).GetDouble().Should().Be(1.1805916207174113e21);
        }

        /// <summary>An empty r reads no root: np.prod's identity at every point (ones), for any rank of r.</summary>
        [TestMethod]
        public void EmptyRoots_GiveOnes()
        {
            Vfr(np.arange(3.0), new object[0]).ToArray<double>().Should().Equal(1.0, 1.0, 1.0);
            var b = Vfr(np.arange(3.0), np.zeros(new Shape(0, 2)));
            b.Should().BeShaped(2, 3);
            b.ToArray<double>().Should().AllBeEquivalentTo(1.0);
            var c = Vfr(new Complex(1, 2), np.zeros(new Shape(0), NPTypeCode.Complex));
            ((Complex)c).Should().Be(Complex.One);
            Vfr(np.zeros(new Shape(0)), new object[] { 1.0, 2.0 }).Should().BeShaped(0);
        }

        /// <summary>
        ///     NumPy's errors in NumPy's order: the ndim check before x - r, the broadcast check, a str operand's no-loop text
        ///     (before any broadcast), CPython's object-loop TypeError at a bare None, CPython's int-to-float overflow.
        /// </summary>
        [TestMethod]
        public void Errors_AreNumPys()
        {
            Action ndim = () => Vfr(np.ones(new Shape(2, 3)), np.ones(new Shape(4, 3)), tensor: false);
            ndim.Should().Throw<ValueError>().WithMessage("x.ndim must be < r.ndim when tensor == False");
            Action bcast = () => Vfr(np.ones(new Shape(3)), np.ones(new Shape(4, 5)), tensor: false);
            bcast.Should().Throw<IncorrectShapeException>().WithMessage("operands could not be broadcast together with shapes (3,) (4,5) ");

            Action xNone = () => Vfr(null, new object[] { 1.0, 2.0 });
            xNone.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'NoneType' and 'float'");
            Action xNoneInt = () => Vfr(null, new object[] { 1L, 2L });   // the roots are float64 by then
            xNoneInt.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'NoneType' and 'float'");
            Action rNone = () => Vfr(np.array(new long[] { 1, 2 }), null);
            rNone.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'int' and 'NoneType'");
            Action rNoneBool = () => Vfr(true, null);
            rNoneBool.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'bool' and 'NoneType'");
            Action both = () => Vfr(null, null);
            both.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'NoneType' and 'NoneType'");
            Action noneStr = () => Vfr(null, "ab");
            noneStr.Should().Throw<TypeError>().WithMessage("unsupported operand type(s) for -: 'NoneType' and 'str'");

            Action xStr = () => Vfr("ab", new object[] { 1.0, 2.0 });
            xStr.Should().Throw<TypeError>()
                .WithMessage("ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U2'), dtype('float64')) -> None");
            Action rStr = () => Vfr(true, "ab");
            rStr.Should().Throw<TypeError>()
                .WithMessage("ufunc 'subtract' did not contain a loop with signature matching types (dtype('bool'), dtype('<U2')) -> None");
            // A str x array meets the str text before the broadcast check (NumPy resolves the loop first).
            Action strFirst = () => Vfr(new object[] { "a", "b", "c" }, np.ones(new Shape(4, 5)), tensor: false);
            strFirst.Should().Throw<TypeError>()
                .WithMessage("ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U1'), dtype('float64')) -> None");

            Action big = () => Vfr(BigInteger.Pow(10, 400), new object[] { 1.0 });
            big.Should().Throw<OverflowException>().WithMessage("int too large to convert to float");
        }

        /// <summary>
        ///     Bare None meeting no element: np.prod of the empty object array is its identity, the Python int 1, for a 0-d
        ///     result — and an object array otherwise, which NumSharp refuses (it has no object dtype).
        /// </summary>
        [TestMethod]
        public void None_WithEmptyOperands()
        {
            var one = Vfr(null, new object[0]);
            one.typecode.Should().Be(NPTypeCode.Int64);
            one.ndim.Should().Be(0);
            one.GetInt64().Should().Be(1L);
            Action objectArray = () => Vfr(null, np.zeros(new Shape(0, 2)));
            objectArray.Should().Throw<NotSupportedException>();
            Action emptyX = () => Vfr(np.zeros(new Shape(0)), null);
            emptyX.Should().Throw<NotSupportedException>();
        }

        /// <summary>
        ///     A Python int past uint64 among the roots makes NumPy's r an object array; at a scalar x NumPy computes with Python
        ///     numbers (the first difference copied, the others multiplied) and returns one — NumSharp the same value 0-d.
        /// </summary>
        [TestMethod]
        public void ObjectRoots_AtAScalarX_AreComputedAsPythonNumbers()
        {
            var big = BigInteger.Pow(2, 70);
            BitConverter.DoubleToInt64Bits(Vfr(1.0, new object[] { big }).GetDouble()).Should().Be(unchecked((long)0xc450000000000000UL));
            var negZero = Vfr(1.0, new object[] { big, 1.0 });
            negZero.GetDouble().Should().Be(0.0);
            double.IsNegative(negZero.GetDouble()).Should().BeTrue();
            var z = (Complex)Vfr(1.0, new object[] { -big, 0.5, new Complex(0, 1.5) });
            Bits(z).Should().Be((0x4440000000000000UL, 0xc448000000000000UL));
            // A NumPy scalar or 0-d array x enters the object loop as its Python value (np.float16 and float32 a Python
            // float, np.uint16 a Python int): float64 results, not the scalar's own dtype.
            foreach (var x in new object[] { (Half)0.5, np.array(0.5f), np.array((Half)0.5) })
            {
                var v = Vfr(x, new object[] { big, 1.5 });
                v.typecode.Should().Be(NPTypeCode.Double);
                v.GetDouble().Should().Be(1.1805916207174113e21);
            }
            Vfr((char)3, new object[] { big, 1.5 }).GetDouble().Should().Be(-1.770887431076117e21);
            // With an array x NumPy's result is an object array.
            Action arrayX = () => Vfr(np.arange(3.0), new object[] { big });
            arrayX.Should().Throw<NotSupportedException>();
        }

        /// <summary>The result is a fresh owning array: writing it never reaches x or r, and it is never one of them.</summary>
        [TestMethod]
        public void Result_IsFreshAndOwning()
        {
            var x = np.arange(4.0);
            var r = np.array(new[] { 1.0, 2.0 });
            var y = Vfr(x, r);
            ReferenceEquals(y, x).Should().BeFalse();
            y.Shape.IsContiguous.Should().BeTrue();
            y[0] = 100.0;
            x.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0);
            r.ToArray<double>().Should().Equal(1.0, 2.0);
        }

        /// <summary>
        ///     The slab-form kernels keep U3's vector lanes for every dtype pair: evaluating every NumPy dtype at 45 points
        ///     (the U-chain, 1-chain vector and tail stages) never falls back to scalar-only chains.
        /// </summary>
        [TestMethod]
        public void SlabKernels_NeverFallBackToScalarChains()
        {
            long before = ILKernelGenerator.PolyVectorFallbacks;
            var codes = new[]
            {
                NPTypeCode.Boolean, NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Int32,
                NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64, NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double,
                NPTypeCode.Complex,
            };
            foreach (var xt in codes)
            {
                var x = (np.arange(45) % 7).astype(xt);
                foreach (var rt in new[] { NPTypeCode.Half, NPTypeCode.Single, NPTypeCode.Double, NPTypeCode.Complex })
                {
                    var r = (np.arange(6) * 0.25).astype(rt);
                    Vfr(x, r).Dispose();
                    Vfr(x, np.broadcast_to(r.reshape(6, 1), new Shape(6, 45)).copy(), tensor: false).Dispose();
                }
            }
            ILKernelGenerator.PolyVectorFallbacks.Should().Be(before);
        }
    }
}
