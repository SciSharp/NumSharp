using System;
using System.Numerics;

namespace NumSharp.Tests.Casting
{
    /// <summary>
    ///     Coverage for the scalar ↔ NDArray cast operators on
    ///     <see cref="NDArray"/>. Ensures NumPy 2.x parity:
    ///     <list type="bullet">
    ///       <item>scalar → NDArray is implicit (always a 0-d array).</item>
    ///       <item>NDArray → number is explicit and requires <c>ndim == 0</c>;
    ///             even single-element 1-d/2-d arrays throw NumPy 2.x's
    ///             <see cref="TypeError"/> "only 0-dimensional arrays can be converted to Python scalars".</item>
    ///       <item>NDArray → <see cref="bool"/> is NumPy's truth test instead: the only element of a one-element array
    ///             of ANY rank, and NumPy's two ValueErrors for a larger or an empty array (parity QA-14).</item>
    ///     </list>
    ///     Focused on <see cref="sbyte"/>, <see cref="Half"/>, and <see cref="Complex"/>
    ///     which were missing cast operators before this branch.
    /// </summary>
    [TestClass]
    public class NDArrayScalarCastTests
    {
        // ---------------------------------------------------------------------
        // Scalar → NDArray (implicit)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Implicit_SByte_To_NDArray()
        {
            NDArray a = (sbyte)42;
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.ndim.Should().Be(0);
            a.size.Should().Be(1);
            ((sbyte)a).Should().Be((sbyte)42);
        }

        [TestMethod]
        public void Implicit_Half_To_NDArray()
        {
            NDArray a = (Half)3.5;
            a.typecode.Should().Be(NPTypeCode.Half);
            a.ndim.Should().Be(0);
            ((Half)a).Should().Be((Half)3.5);
        }

        [TestMethod]
        public void Implicit_Complex_To_NDArray()
        {
            NDArray a = new Complex(1, 2);
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.ndim.Should().Be(0);
            ((Complex)a).Should().Be(new Complex(1, 2));
        }

        // ---------------------------------------------------------------------
        // NDArray → scalar (explicit, 0-d only)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Explicit_SByte_From_ZeroD() =>
            ((sbyte)NDArray.Scalar<sbyte>(42)).Should().Be((sbyte)42);

        [TestMethod]
        public void Explicit_Half_From_ZeroD() =>
            ((Half)NDArray.Scalar<Half>((Half)2.5)).Should().Be((Half)2.5);

        [TestMethod]
        public void Explicit_Complex_From_ZeroD() =>
            ((Complex)NDArray.Scalar<Complex>(new Complex(7, 3))).Should().Be(new Complex(7, 3));

        // ---------------------------------------------------------------------
        // Boundary values
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Boundary_SByte_MaxValue() =>
            ((sbyte)NDArray.Scalar<sbyte>(sbyte.MaxValue)).Should().Be(sbyte.MaxValue);

        [TestMethod]
        public void Boundary_SByte_MinValue() =>
            ((sbyte)NDArray.Scalar<sbyte>(sbyte.MinValue)).Should().Be(sbyte.MinValue);

        [TestMethod]
        public void Boundary_Half_NaN_Preserved() =>
            Half.IsNaN((Half)NDArray.Scalar<Half>(Half.NaN)).Should().BeTrue();

        [TestMethod]
        public void Boundary_Half_PosInf_Preserved() =>
            Half.IsPositiveInfinity((Half)NDArray.Scalar<Half>(Half.PositiveInfinity)).Should().BeTrue();

        [TestMethod]
        public void Boundary_Half_NegInf_Preserved() =>
            Half.IsNegativeInfinity((Half)NDArray.Scalar<Half>(Half.NegativeInfinity)).Should().BeTrue();

        [TestMethod]
        public void Boundary_Half_MaxValue_Preserved() =>
            ((Half)NDArray.Scalar<Half>(Half.MaxValue)).Should().Be(Half.MaxValue);

        [TestMethod]
        public void Boundary_Half_MinValue_Preserved() =>
            ((Half)NDArray.Scalar<Half>(Half.MinValue)).Should().Be(Half.MinValue);

        [TestMethod]
        public void Boundary_Complex_Zero() =>
            ((Complex)NDArray.Scalar<Complex>(Complex.Zero)).Should().Be(Complex.Zero);

        [TestMethod]
        public void Boundary_Complex_One() =>
            ((Complex)NDArray.Scalar<Complex>(Complex.One)).Should().Be(Complex.One);

        [TestMethod]
        public void Boundary_Complex_ImaginaryOne() =>
            ((Complex)NDArray.Scalar<Complex>(Complex.ImaginaryOne)).Should().Be(Complex.ImaginaryOne);

        [TestMethod]
        public void Boundary_Complex_Negative() =>
            ((Complex)NDArray.Scalar<Complex>(new Complex(-3, -4))).Should().Be(new Complex(-3, -4));

        // ---------------------------------------------------------------------
        // Cross-type conversion via Converts.ChangeType
        // ---------------------------------------------------------------------

        [TestMethod]
        public void CrossType_Int32_To_Half() =>
            ((Half)NDArray.Scalar<int>(42)).Should().Be((Half)42);

        [TestMethod]
        public void CrossType_Double_To_Half()
        {
            var result = (Half)NDArray.Scalar<double>(3.14);
            // Half has ~3 sig-digit precision near 3.14: expect 3.140625
            System.Math.Abs((float)result - 3.14f).Should().BeLessThan(0.01f);
        }

        [TestMethod]
        public void CrossType_Int32_To_Complex()
        {
            var result = (Complex)NDArray.Scalar<int>(42);
            result.Real.Should().Be(42);
            result.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void CrossType_Half_To_SByte() =>
            ((sbyte)NDArray.Scalar<Half>((Half)7.5)).Should().Be((sbyte)7);

        /// <summary>NumPy 2.4.2: <c>float(np.array(3.5+1.7j))</c> raises CPython's <c>float()</c> TypeError; NumSharp has no
        ///     ComplexWarning, so a scalar cast refuses rather than dropping the imaginary part (use <c>np.real(nd)</c>).</summary>
        [TestMethod]
        public void CrossType_Complex_To_Half_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(3.5, 1.7));
            Action act = () => { var _ = (Half)nd; };
            act.Should().Throw<TypeError>().WithMessage("float() argument must be a string or a real number, not 'complex'");
        }

        [TestMethod]
        public void CrossType_Complex_To_Half_Throws_Even_IfImaginaryZero()
        {
            // NumPy's rule applies even when imaginary == 0: int(np.complex128(3+0j)) throws.
            var nd = NDArray.Scalar<Complex>(new Complex(3.5, 0));
            Action act = () => { var _ = (Half)nd; };
            act.Should().Throw<TypeError>();
        }

        /// <summary>NumPy 2.4.2: <c>int(np.array(3+4j))</c> raises CPython's <c>int()</c> TypeError, text for text.</summary>
        [TestMethod]
        public void CrossType_Complex_To_Int_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(3, 4));
            Action act = () => { var _ = (int)nd; };
            act.Should().Throw<TypeError>().WithMessage(
                "int() argument must be a string, a bytes-like object or a real number, not 'complex'");
        }

        [TestMethod]
        public void CrossType_Complex_To_Double_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(3, 4));
            Action act = () => { var _ = (double)nd; };
            act.Should().Throw<TypeError>();
        }

        [TestMethod]
        public void CrossType_Complex_To_SByte_Throws_TypeError()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(5, 2));
            Action act = () => { var _ = (sbyte)nd; };
            act.Should().Throw<TypeError>();
        }

        /// <summary>NumPy 2.4.2: <c>bool(np.array(1+0j))</c> is True — a complex 0-d array converts to its truth (either part
        ///     nonzero), not a TypeError (parity QA-42).</summary>
        [TestMethod]
        public void CrossType_Complex_To_Bool_IsTheTruth()
        {
            var nd = NDArray.Scalar<Complex>(new Complex(1, 0));
            ((bool)nd).Should().BeTrue();
            var zero = NDArray.Scalar<Complex>(Complex.Zero);
            ((bool)zero).Should().BeFalse();
        }

        [TestMethod]
        public void CrossType_Complex_To_Complex_Works()
        {
            // Complex → Complex is still allowed (no conversion needed).
            var nd = NDArray.Scalar<Complex>(new Complex(3, 4));
            ((Complex)nd).Should().Be(new Complex(3, 4));
        }

        [TestMethod]
        public void CrossType_SByte_To_Complex_ImagIsZero()
        {
            var nd = NDArray.Scalar<sbyte>(-42);
            var result = (Complex)nd;
            result.Real.Should().Be(-42);
            result.Imaginary.Should().Be(0);
        }

        // ---------------------------------------------------------------------
        // ndim != 0 must throw (NumPy 2.x strict)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void OneD_NDArray_Cast_To_SByte_Throws()
        {
            var arr = np.array(new sbyte[] { 1, 2, 3 });
            Action act = () => { var _ = (sbyte)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void OneD_NDArray_Cast_To_Half_Throws()
        {
            var arr = np.array(new Half[] { (Half)1.0, (Half)2.0 });
            Action act = () => { var _ = (Half)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void OneD_NDArray_Cast_To_Complex_Throws()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2), new Complex(3, 4) });
            Action act = () => { var _ = (Complex)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void OneD_SingleElement_Still_Throws_SByte()
        {
            // NumPy 2.x: np.array([42], dtype=int8) -> int(x) raises TypeError
            var arr = np.array(new sbyte[] { 42 });
            Action act = () => { var _ = (sbyte)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void OneD_SingleElement_Still_Throws_Half()
        {
            var arr = np.array(new Half[] { (Half)3.5 });
            Action act = () => { var _ = (Half)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void OneD_SingleElement_Still_Throws_Complex()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2) });
            Action act = () => { var _ = (Complex)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void TwoD_OneByOne_Still_Throws_SByte()
        {
            // NumPy 2.x: np.array([[42]], dtype=int8) -> int(x) raises TypeError
            var arr = np.array(new sbyte[] { 42 }).reshape(1, 1);
            Action act = () => { var _ = (sbyte)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void TwoD_OneByOne_Still_Throws_Half()
        {
            var arr = np.array(new Half[] { (Half)3.5 }).reshape(1, 1);
            Action act = () => { var _ = (Half)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        [TestMethod]
        public void TwoD_OneByOne_Still_Throws_Complex()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2) }).reshape(1, 1);
            Action act = () => { var _ = (Complex)arr; };
            act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }

        // ---------------------------------------------------------------------
        // Round-trip via indexing (arr[i] returns 0-d NDArray)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Indexing_SByte_RoundTrip()
        {
            var arr = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            ((sbyte)arr[0]).Should().Be((sbyte)(-128));
            ((sbyte)arr[1]).Should().Be((sbyte)(-1));
            ((sbyte)arr[2]).Should().Be((sbyte)0);
            ((sbyte)arr[3]).Should().Be((sbyte)1);
            ((sbyte)arr[4]).Should().Be((sbyte)127);
        }

        [TestMethod]
        public void Indexing_Half_RoundTrip()
        {
            var arr = np.array(new Half[] { Half.MinValue, (Half)(-1.5), Half.Zero, (Half)1.5, Half.MaxValue });
            ((Half)arr[0]).Should().Be(Half.MinValue);
            ((Half)arr[1]).Should().Be((Half)(-1.5));
            ((Half)arr[2]).Should().Be(Half.Zero);
            ((Half)arr[3]).Should().Be((Half)1.5);
            ((Half)arr[4]).Should().Be(Half.MaxValue);
        }

        [TestMethod]
        public void Indexing_Complex_RoundTrip()
        {
            var arr = np.array(new Complex[] { new Complex(1, 2), new Complex(-3, -4), Complex.Zero, Complex.One, Complex.ImaginaryOne });
            ((Complex)arr[0]).Should().Be(new Complex(1, 2));
            ((Complex)arr[1]).Should().Be(new Complex(-3, -4));
            ((Complex)arr[2]).Should().Be(Complex.Zero);
            ((Complex)arr[3]).Should().Be(Complex.One);
            ((Complex)arr[4]).Should().Be(Complex.ImaginaryOne);
        }

        [TestMethod]
        public void Indexing_TwoD_SByte_RoundTrip()
        {
            var arr = np.array(new sbyte[] { 1, 2, 3, 4, 5, 6 }).reshape(2, 3);
            ((sbyte)arr[1, 2]).Should().Be((sbyte)6);
            ((sbyte)arr[0, 0]).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void Indexing_TwoD_Half_RoundTrip()
        {
            var arr = np.array(new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 }).reshape(2, 2);
            ((Half)arr[0, 1]).Should().Be((Half)2);
            ((Half)arr[1, 0]).Should().Be((Half)3);
        }

        // ---------------------------------------------------------------------
        // Implicit scalar conversions compose with operations
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Implicit_SByte_Used_In_Arithmetic()
        {
            NDArray a = (sbyte)10;
            NDArray b = (sbyte)5;
            // sbyte + sbyte → int8 (NumPy: same dtype preserved for same-kind)
            var sum = a + b;
            sum.typecode.Should().Be(NPTypeCode.SByte);
            ((sbyte)sum).Should().Be((sbyte)15);
        }

        [TestMethod]
        public void Implicit_Half_Used_In_Arithmetic()
        {
            NDArray a = (Half)10;
            NDArray b = (Half)5;
            var sum = a + b;
            sum.typecode.Should().Be(NPTypeCode.Half);
            ((Half)sum).Should().Be((Half)15);
        }

        [TestMethod]
        public void Implicit_Complex_Used_In_Arithmetic()
        {
            NDArray a = new Complex(3, 4);
            NDArray b = new Complex(1, 2);
            var sum = a + b;
            sum.typecode.Should().Be(NPTypeCode.Complex);
            ((Complex)sum).Should().Be(new Complex(4, 6));
        }

        // ================================================================
        // Python's int(), checked (parity QA-42; NumPy 2.4.2's answers, LP64 texts)
        // ================================================================

        /// <summary>
        ///     NumPy's <c>int(arr)</c> of a float NaN is CPython's <c>ValueError: cannot convert float NaN to integer</c> and of
        ///     an infinity <c>OverflowError: cannot convert float infinity to integer</c>, for every float lane (float16 too);
        ///     the integer conversions used to return <c>MinValue</c> silently.
        /// </summary>
        [TestMethod]
        public void IntegerTargets_NaNAndInfinity_RaiseLikeInt()
        {
            using var nan = np.array(double.NaN);
            using var inf = np.array(double.PositiveInfinity);
            using var negInf = NDArray.Scalar(float.NegativeInfinity);
            using var halfNaN = NDArray.Scalar(Half.NaN);

            Action nanToInt = () => { int _ = (int)nan; };
            nanToInt.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            Action halfNaNToLong = () => { long _ = (long)halfNaN; };
            halfNaNToLong.Should().Throw<ValueError>().WithMessage("cannot convert float NaN to integer");
            Action infToInt = () => { int _ = (int)inf; };
            infToInt.Should().Throw<OverflowException>().WithMessage("cannot convert float infinity to integer");
            Action negInfToByte = () => { byte _ = (byte)negInf; };
            negInfToByte.Should().Throw<OverflowException>().WithMessage("cannot convert float infinity to integer");
        }

        /// <summary>
        ///     NumPy's <c>int()</c> is exact (<c>int(np.array(3e9))</c> is 3000000000, <c>int(np.array(2**64-1))</c> is
        ///     18446744073709551615), so a value the C# type cannot hold raises with NumPy's text for packing that Python int
        ///     into the dtype (LP64, as <c>NDExprTypeRules.CheckPyIntPacksIntoLoop</c> reports it): <c>Python integer 3000000000
        ///     out of bounds for int32</c>; past the 64-bit C long (into any type but uint32 / uint64) <c>Python int too large
        ///     to convert to C long</c>. They used to wrap (<c>int.MinValue</c>, <c>-1</c>, <c>long.MinValue</c>, 255).
        /// </summary>
        [TestMethod]
        public void IntegerTargets_OutOfRange_RaiseInsteadOfWrapping()
        {
            using var threeE9 = np.array(3e9);
            using var u64Max = np.array(ulong.MaxValue);
            using var huge = np.array(1e300);
            using var minusOne = np.array(-1L);
            using var hugeForU64 = np.array(1e20);

            Action toInt = () => { int _ = (int)threeE9; };
            toInt.Should().Throw<OverflowException>().WithMessage("Python integer 3000000000 out of bounds for int32");
            Action toLong = () => { long _ = (long)u64Max; };
            toLong.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            Action hugeToLong = () => { long _ = (long)huge; };
            hugeToLong.Should().Throw<OverflowException>().WithMessage("Python int too large to convert to C long");
            Action negativeToByte = () => { byte _ = (byte)minusOne; };
            negativeToByte.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint8");
            Action negativeToULong = () => { ulong _ = (ulong)minusOne; };
            negativeToULong.Should().Throw<OverflowException>().WithMessage("Python integer -1 out of bounds for uint64");
            Action pastU64 = () => { ulong _ = (ulong)hugeForU64; };
            pastU64.Should().Throw<OverflowException>().WithMessage("Python integer 100000000000000000000 out of bounds for uint64");
        }

        /// <summary>
        ///     In range, <c>int()</c> truncates toward zero and is exact: <c>(int)np.array(-3.7)</c> is -3, <c>-0.7</c> gives 0,
        ///     uint64 max fits <c>(ulong)</c>, 2^53 + 2 (a double) fits <c>(long)</c> exactly, a bool is 0 / 1, a float16 and a
        ///     decimal truncate, a text is parsed by CPython's <c>int()</c> (<c>'3000000000'</c> fits a long).
        /// </summary>
        [TestMethod]
        public void IntegerTargets_InRange_TruncateExactly()
        {
            using var a = np.array(-3.7);
            ((int)a).Should().Be(-3);
            using var b = np.array(-0.7);
            ((int)b).Should().Be(0);
            using var c = np.array(ulong.MaxValue);
            ((ulong)c).Should().Be(ulong.MaxValue);
            using var d = np.array(9007199254740994.0);
            ((long)d).Should().Be(9007199254740994L);
            using var e = np.array(true);
            ((int)e).Should().Be(1);
            using var f = NDArray.Scalar((Half)(-2.5));
            ((short)f).Should().Be((short)-2);
            using var g = NDArray.Scalar(7.9m);
            ((byte)g).Should().Be((byte)7);
            using var h = np.array("3000000000");
            ((long)h).Should().Be(3000000000L);
            using var i = np.array(65535L);
            ((char)i).Should().Be((char)65535);
        }

        /// <summary>
        ///     NumPy's <c>bool(arr)</c> is the element's truth for every kind: a NaN is True, <c>-0.0</c> False, a complex True
        ///     when either part is nonzero; and <c>float(arr)</c> of an integer is exact up to 2^53, of a huge float32 target
        ///     ±inf (the C# float is NumPy's float32 of the Python float).
        /// </summary>
        [TestMethod]
        public void BoolAndFloatTargets_KeepPythonsTruthAndFloat()
        {
            using var nan = np.array(double.NaN);
            ((bool)nan).Should().BeTrue();
            using var negZero = np.array(-0.0);
            ((bool)negZero).Should().BeFalse();
            using var imaginary = np.array(new Complex(0, 1));
            ((bool)imaginary).Should().BeTrue();
            using var big = np.array(1e300);
            ((float)big).Should().Be(float.PositiveInfinity);
            using var exact = np.array(9007199254740992L);
            ((double)exact).Should().Be(9007199254740992.0);
        }

        // ================================================================
        // Python's bool(): the truth test of ANY one-element array (parity QA-14; NumPy 2.4.2's answers)
        // ================================================================

        /// <summary>
        ///     NumPy's <c>bool(arr)</c> is <c>array_nonzero</c>, which reads the only element of a one-element array whatever
        ///     its rank: <c>bool(np.array([5.0]))</c> is True, <c>bool(np.array([[0.0]]))</c> False, a NaN True, <c>-0.0</c>
        ///     False, a complex True when either part is nonzero, a text True when non-empty. The cast used to refuse every
        ///     N-D array with the numeric conversions' 0-d rule.
        /// </summary>
        [TestMethod]
        public void Bool_OneElementOfAnyRank_IsTheElementsTruth()
        {
            using var five = np.array(new[] { 5.0 });
            using var zero2D = np.array(new[,] { { 0.0 } });
            using var nan = np.array(new[] { double.NaN });
            using var negZero = np.array(new[] { -0.0 });
            using var complexZero3D = np.array(new[] { Complex.Zero }).reshape(1, 1, 1);
            using var imaginary = np.array(new[] { Complex.ImaginaryOne });
            using var text = np.array(new[] { "a" });
            using var emptyText = np.array(new[] { "" });
            using var truth = np.array(new[] { true });
            using var int8Zero = np.array(new sbyte[,] { { 0 } });
            using var half = np.array(new[] { (Half)0.5 });

            ((bool)five).Should().BeTrue();
            ((bool)zero2D).Should().BeFalse();
            ((bool)nan).Should().BeTrue();
            ((bool)negZero).Should().BeFalse();
            ((bool)complexZero3D).Should().BeFalse();
            ((bool)imaginary).Should().BeTrue();
            ((bool)text).Should().BeTrue();
            ((bool)emptyText).Should().BeFalse();
            ((bool)truth).Should().BeTrue();
            ((bool)int8Zero).Should().BeFalse();
            ((bool)half).Should().BeTrue();
        }

        /// <summary>
        ///     The truth test reads the view's own element, not its base's first: <c>arange(6).reshape(2, 3)[1:2, 2:3]</c> is
        ///     <c>[[5]]</c> (True) at a non-zero offset, <c>[0:1, 0:1]</c> is <c>[[0]]</c> (False), and a <c>(1, 1)</c>
        ///     broadcast of a 0-d zero is False.
        /// </summary>
        [TestMethod]
        public void Bool_OneElementView_ReadsTheViewsElement()
        {
            using var grid = np.arange(6).reshape(2, 3);
            using var five = grid["1:2, 2:3"];
            using var zero = grid["0:1, 0:1"];
            using var broadcastZero = np.broadcast_to(np.array(0.0), new Shape(1, 1));

            ((bool)five).Should().BeTrue();
            ((bool)zero).Should().BeFalse();
            ((bool)broadcastZero).Should().BeFalse();
        }

        /// <summary>
        ///     More than one element is ambiguous whatever the dtype: NumPy's <c>ValueError: The truth value of an array with
        ///     more than one element is ambiguous. Use a.any() or a.all()</c> (the size check comes before any element check,
        ///     so a complex array gets it too).
        /// </summary>
        [TestMethod]
        public void Bool_MoreThanOneElement_RaisesTheAmbiguityValueError()
        {
            using var two = np.array(new[] { 1.0, 2.0 });
            using var twoComplex = np.array(new[] { Complex.ImaginaryOne, new Complex(0, 2) });

            Action real = () => _ = (bool)two;
            Action complex = () => _ = (bool)twoComplex;

            real.Should().Throw<ValueError>()
                .WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
            complex.Should().Throw<ValueError>()
                .WithMessage("The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()");
        }

        /// <summary>
        ///     An empty array is ambiguous too (NumPy 2.x made it an error): <c>ValueError: The truth value of an empty array
        ///     is ambiguous. Use `array.size &gt; 0` to check that an array is not empty.</c>, for <c>(0,)</c> and
        ///     <c>(0, 3)</c> alike.
        /// </summary>
        [TestMethod]
        public void Bool_Empty_RaisesTheEmptyValueError()
        {
            using var empty = np.array(Array.Empty<double>());
            using var emptyRows = np.zeros(new Shape(0, 3));

            Action flat = () => _ = (bool)empty;
            Action rows = () => _ = (bool)emptyRows;

            flat.Should().Throw<ValueError>().WithMessage(
                "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
            rows.Should().Throw<ValueError>().WithMessage(
                "The truth value of an empty array is ambiguous. Use `array.size > 0` to check that an array is not empty.");
        }

        /// <summary>
        ///     <c>int()</c> / <c>float()</c> / <c>complex()</c> keep NumPy 2.x's 0-d rule, with NumPy's type and text:
        ///     <c>TypeError: only 0-dimensional arrays can be converted to Python scalars</c> for a one-element 1-D or 2-D
        ///     array, an empty array, a larger one and a text array; the rank check comes first, so <c>int(np.array([1j]))</c>
        ///     gets it rather than the complex-to-int refusal. They threw <see cref="IncorrectShapeException"/> with
        ///     NumSharp's own words.
        /// </summary>
        [TestMethod]
        public void NumericTargets_NonZeroDim_RaiseNumPysTypeError()
        {
            using var oneFloat = np.array(new[] { 5.5 });
            using var oneInt2D = np.array(new long[,] { { 3 } });
            using var oneComplex = np.array(new[] { Complex.ImaginaryOne });
            using var empty = np.array(Array.Empty<double>());
            using var square = np.zeros(new Shape(2, 2));
            using var text = np.array(new[] { "1.5" });

            Action toDouble = () => _ = (double)oneFloat;
            Action toLong = () => _ = (long)oneInt2D;
            Action toComplex = () => _ = (Complex)oneComplex;
            Action complexToInt = () => _ = (int)oneComplex;
            Action emptyToDouble = () => _ = (double)empty;
            Action squareToInt = () => _ = (int)square;
            Action textToDouble = () => _ = (double)text;

            foreach (var act in new[] { toDouble, toLong, toComplex, complexToInt, emptyToDouble, squareToInt, textToDouble })
                act.Should().Throw<TypeError>().WithMessage("only 0-dimensional arrays can be converted to Python scalars");
        }
    }
}
