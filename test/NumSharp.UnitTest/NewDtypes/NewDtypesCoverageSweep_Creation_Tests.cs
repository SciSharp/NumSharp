using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Round 11 — Coverage sweep of every Creation API for Half / Complex / SByte,
    /// battletested against NumPy 2.4.2 (189-case matrix, 100% parity after B27/B28/B29 fixes).
    ///
    /// Bugs closed in this round:
    ///   B27 — np.eye(N, M, k) used wrong stride (N+1 instead of M+1) for non-square
    ///         matrices or k != 0. Affected ALL dtypes, not just the new ones.
    ///   B28 — np.asanyarray(NDArray, Type dtype) ignored dtype on the NDArray fast-path.
    ///         The bottom `astype` call was unreachable for NDArray inputs.
    ///   B29 — np.asarray(NDArray, Type dtype) overload was missing (API gap vs NumPy).
    ///
    /// Verified against:
    ///   python -c "import numpy as np; print(np.eye(4,3,dtype=np.float16))"
    ///   python -c "import numpy as np; print(np.asanyarray(np.zeros((2,3)), dtype=np.float16))"
    /// </summary>
    [TestClass]
    public class NewDtypesCoverageSweep_Creation_Tests
    {
        private const double HalfTol = 1e-3;
        private const double CplxTol = 1e-12;

        private static Complex C(double r, double i) => new Complex(r, i);

        #region zeros / ones / empty — all 3 dtypes, all shape variants

        [TestMethod]
        public void Zeros_Half_1D()
        {
            var a = np.zeros(new Shape(5), typeof(Half));
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 5 });
            for (int i = 0; i < 5; i++)
                a.GetAtIndex<Half>(i).Should().Be((Half)0);
        }

        [TestMethod]
        public void Zeros_Complex_2D()
        {
            var a = np.zeros(new Shape(2, 3), typeof(Complex));
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.shape.Should().Equal(new long[] { 2, 3 });
            for (int i = 0; i < 6; i++)
                a.GetAtIndex<Complex>(i).Should().Be(C(0, 0));
        }

        [TestMethod]
        public void Zeros_SByte_3D()
        {
            var a = np.zeros(new Shape(2, 3, 4), typeof(sbyte));
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.shape.Should().Equal(new long[] { 2, 3, 4 });
            a.size.Should().Be(24);
            for (int i = 0; i < 24; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)0);
        }

        [TestMethod]
        public void Zeros_Empty_Half() => np.zeros(new Shape(0), typeof(Half)).size.Should().Be(0);
        [TestMethod]
        public void Zeros_Empty_Complex() => np.zeros(new Shape(0, 5), typeof(Complex)).size.Should().Be(0);
        [TestMethod]
        public void Zeros_Empty_SByte() => np.zeros(new Shape(0), typeof(sbyte)).size.Should().Be(0);

        [TestMethod]
        public void Ones_Half_1D()
        {
            var a = np.ones(new Shape(5), typeof(Half));
            for (int i = 0; i < 5; i++)
                a.GetAtIndex<Half>(i).Should().Be((Half)1);
        }

        [TestMethod]
        public void Ones_Complex_1D()
        {
            var a = np.ones(new Shape(5), typeof(Complex));
            for (int i = 0; i < 5; i++)
                a.GetAtIndex<Complex>(i).Should().Be(C(1, 0));
        }

        [TestMethod]
        public void Ones_SByte_1D()
        {
            var a = np.ones(new Shape(5), typeof(sbyte));
            for (int i = 0; i < 5; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void Empty_Half_ReturnsCorrectShapeAndDtype()
        {
            var a = np.empty(new Shape(3, 4), typeof(Half));
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 3, 4 });
            a.size.Should().Be(12);
        }

        [TestMethod]
        public void Empty_Complex_ReturnsCorrectShapeAndDtype()
        {
            var a = np.empty(new Shape(3, 4), typeof(Complex));
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.size.Should().Be(12);
        }

        [TestMethod]
        public void Empty_SByte_ReturnsCorrectShapeAndDtype()
        {
            var a = np.empty(new Shape(3, 4), typeof(sbyte));
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.size.Should().Be(12);
        }

        #endregion

        #region full — fill value preservation across dtypes

        [TestMethod]
        public void Full_Half_TypicalValue()
        {
            var a = np.full(new Shape(3), (Half)1.5, typeof(Half));
            for (int i = 0; i < 3; i++)
                a.GetAtIndex<Half>(i).Should().Be((Half)1.5);
        }

        [TestMethod]
        public void Full_Half_MaxFinite_65504()
        {
            var a = np.full(new Shape(3), (Half)65504, typeof(Half));
            ((double)a.GetAtIndex<Half>(0)).Should().Be(65504.0);
        }

        [TestMethod]
        public void Full_Half_Infinity()
        {
            var a = np.full(new Shape(3), Half.PositiveInfinity, typeof(Half));
            Half.IsPositiveInfinity(a.GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Full_Half_NaN()
        {
            var a = np.full(new Shape(3), Half.NaN, typeof(Half));
            Half.IsNaN(a.GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        public void Full_Complex_ImaginaryPreserved()
        {
            var a = np.full(new Shape(3), C(1, 2), typeof(Complex));
            for (int i = 0; i < 3; i++)
                a.GetAtIndex<Complex>(i).Should().Be(C(1, 2));
        }

        [TestMethod]
        public void Full_Complex_WithInfinityReal()
        {
            var a = np.full(new Shape(3), C(double.PositiveInfinity, 0), typeof(Complex));
            var v = a.GetAtIndex<Complex>(0);
            double.IsPositiveInfinity(v.Real).Should().BeTrue();
            v.Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Full_SByte_Min()
        {
            var a = np.full(new Shape(3), (sbyte)(-128), typeof(sbyte));
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-128));
        }

        [TestMethod]
        public void Full_SByte_Max()
        {
            var a = np.full(new Shape(3), (sbyte)127, typeof(sbyte));
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)127);
        }

        #endregion

        #region arange

        [TestMethod]
        public void Arange_Half_PositiveStep()
        {
            var a = np.arange(0.0, 5.0, 1.0, NPTypeCode.Half);
            a.size.Should().Be(5);
            for (int i = 0; i < 5; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().BeApproximately((double)i, HalfTol);
        }

        [TestMethod]
        public void Arange_Half_FractionalStep()
        {
            // np.arange(0, 5, 0.5, dtype=float16) -> 10 elements
            var a = np.arange(0.0, 5.0, 0.5, NPTypeCode.Half);
            a.size.Should().Be(10);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(0.0);
            ((double)a.GetAtIndex<Half>(1)).Should().BeApproximately(0.5, HalfTol);
            ((double)a.GetAtIndex<Half>(9)).Should().BeApproximately(4.5, HalfTol);
        }

        [TestMethod]
        public void Arange_Half_NegativeStep()
        {
            var a = np.arange(5.0, 0.0, -1.0, NPTypeCode.Half);
            a.size.Should().Be(5);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(5.0);
            ((double)a.GetAtIndex<Half>(4)).Should().Be(1.0);
        }

        [TestMethod]
        public void Arange_Half_Empty()
        {
            var a = np.arange(1.0, 1.0, 1.0, NPTypeCode.Half);
            a.size.Should().Be(0);
            a.typecode.Should().Be(NPTypeCode.Half);
        }

        [TestMethod]
        public void Arange_Complex_PositiveStep()
        {
            var a = np.arange(0.0, 5.0, 1.0, NPTypeCode.Complex);
            a.size.Should().Be(5);
            for (int i = 0; i < 5; i++)
            {
                var v = a.GetAtIndex<Complex>(i);
                v.Real.Should().BeApproximately(i, CplxTol);
                v.Imaginary.Should().Be(0);
            }
        }

        [TestMethod]
        public void Arange_SByte_PositiveStep()
        {
            var a = np.arange(0.0, 10.0, 1.0, NPTypeCode.SByte);
            a.size.Should().Be(10);
            for (int i = 0; i < 10; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)i);
        }

        [TestMethod]
        public void Arange_SByte_NegativeStep()
        {
            var a = np.arange(10.0, -10.0, -2.0, NPTypeCode.SByte);
            a.size.Should().Be(10);
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)10);
            a.GetAtIndex<sbyte>(9).Should().Be((sbyte)(-8));
        }

        [TestMethod]
        public void Arange_SByte_BoundaryValues()
        {
            var a = np.arange(-128.0, 127.0, 50.0, NPTypeCode.SByte);
            a.size.Should().Be(6);
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-128));
            a.GetAtIndex<sbyte>(5).Should().Be((sbyte)122);
        }

        [TestMethod]
        public void Arange_SByte_Empty()
        {
            var a = np.arange(0.0, 0.0, 1.0, NPTypeCode.SByte);
            a.size.Should().Be(0);
            a.typecode.Should().Be(NPTypeCode.SByte);
        }

        #endregion

        #region linspace

        [TestMethod]
        public void Linspace_Half_Endpoint()
        {
            var a = np.linspace(0.0, 1.0, 5L, true, NPTypeCode.Half);
            a.size.Should().Be(5);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(0.0);
            ((double)a.GetAtIndex<Half>(4)).Should().Be(1.0);
            ((double)a.GetAtIndex<Half>(2)).Should().BeApproximately(0.5, HalfTol);
        }

        [TestMethod]
        public void Linspace_Half_NoEndpoint()
        {
            var a = np.linspace(0.0, 1.0, 5L, false, NPTypeCode.Half);
            a.size.Should().Be(5);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(0.0);
            ((double)a.GetAtIndex<Half>(4)).Should().BeApproximately(0.8, HalfTol);
        }

        [TestMethod]
        public void Linspace_Complex_Endpoint()
        {
            var a = np.linspace(-5.0, 5.0, 11L, true, NPTypeCode.Complex);
            a.size.Should().Be(11);
            a.GetAtIndex<Complex>(0).Should().Be(C(-5, 0));
            a.GetAtIndex<Complex>(10).Should().Be(C(5, 0));
        }

        [TestMethod]
        public void Linspace_SByte_Endpoint()
        {
            var a = np.linspace(0.0, 10.0, 11L, true, NPTypeCode.SByte);
            a.size.Should().Be(11);
            for (int i = 0; i < 11; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)i);
        }

        [TestMethod]
        public void Linspace_Half_SingleElement_ReturnsStart()
        {
            var a = np.linspace(0.0, 1.0, 1L, true, NPTypeCode.Half);
            a.size.Should().Be(1);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(0.0);
        }

        [TestMethod]
        public void Linspace_Complex_Zero_Empty()
        {
            var a = np.linspace(0.0, 1.0, 0L, true, NPTypeCode.Complex);
            a.size.Should().Be(0);
            a.typecode.Should().Be(NPTypeCode.Complex);
        }

        #endregion

        #region eye / identity — B27 regression

        [TestMethod]
        public void B27_Eye_Half_Square()
        {
            // np.eye(3, dtype=float16) → 3×3 identity
            var a = np.eye(3, dtype: typeof(Half));
            a.shape.Should().Equal(new long[] { 3, 3 });
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    ((double)a.GetAtIndex<Half>((long)r * 3 + c)).Should().Be(r == c ? 1.0 : 0.0);
        }

        [TestMethod]
        public void B27_Eye_Half_Rectangular_4x3()
        {
            // np.eye(4,3,dtype=float16) main diagonal at indices 0,4,8 (M+1 stride, not N+1).
            // NumPy: [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
            var a = np.eye(4, 3, 0, typeof(Half));
            a.shape.Should().Equal(new long[] { 4, 3 });
            var expected = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };
            for (int i = 0; i < 12; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(expected[i], $"index {i}");
        }

        [TestMethod]
        public void B27_Eye_Complex_Rectangular_4x3()
        {
            var a = np.eye(4, 3, 0, typeof(Complex));
            var expected = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };
            for (int i = 0; i < 12; i++)
                a.GetAtIndex<Complex>(i).Should().Be(C(expected[i], 0), $"index {i}");
        }

        [TestMethod]
        public void B27_Eye_SByte_Rectangular_4x3()
        {
            var a = np.eye(4, 3, 0, typeof(sbyte));
            var expected = new sbyte[] { 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0 };
            for (int i = 0; i < 12; i++)
                a.GetAtIndex<sbyte>(i).Should().Be(expected[i], $"index {i}");
        }

        [TestMethod]
        public void B27_Eye_Half_UpperDiagonal_3x4_k1()
        {
            // np.eye(3,4,k=1,dtype=float16): [[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            var a = np.eye(3, 4, 1, typeof(Half));
            var expected = new double[] { 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
            for (int i = 0; i < 12; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(expected[i], $"index {i}");
        }

        [TestMethod]
        public void B27_Eye_Half_LowerDiagonal_3x4_kNeg1()
        {
            // np.eye(3,4,k=-1,dtype=float16): [[0,0,0,0],[1,0,0,0],[0,1,0,0]]
            var a = np.eye(3, 4, -1, typeof(Half));
            var expected = new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 };
            for (int i = 0; i < 12; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(expected[i], $"index {i}");
        }

        [TestMethod]
        public void B27_Eye_SByte_UpperDiagonal_3x4_k1()
        {
            var a = np.eye(3, 4, 1, typeof(sbyte));
            var expected = new sbyte[] { 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
            for (int i = 0; i < 12; i++)
                a.GetAtIndex<sbyte>(i).Should().Be(expected[i], $"index {i}");
        }

        [TestMethod]
        public void B27_Eye_SByte_LowerDiagonal_3x4_kNeg1()
        {
            var a = np.eye(3, 4, -1, typeof(sbyte));
            var expected = new sbyte[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0 };
            for (int i = 0; i < 12; i++)
                a.GetAtIndex<sbyte>(i).Should().Be(expected[i], $"index {i}");
        }

        [TestMethod]
        public void Eye_KOutsideMatrix_ReturnsZeros_Half()
        {
            var a = np.eye(3, 3, 5, typeof(Half));
            a.size.Should().Be(9);
            for (int i = 0; i < 9; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(0.0);
        }

        [TestMethod]
        public void Eye_ZeroSize_Half()
        {
            var a = np.eye(0, 0, 0, typeof(Half));
            a.size.Should().Be(0);
        }

        [TestMethod]
        public void Identity_Half()
        {
            var a = np.identity(3, typeof(Half));
            a.shape.Should().Equal(new long[] { 3, 3 });
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    ((double)a.GetAtIndex<Half>((long)r * 3 + c)).Should().Be(r == c ? 1.0 : 0.0);
        }

        [TestMethod]
        public void Identity_Complex()
        {
            var a = np.identity(3, typeof(Complex));
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    a.GetAtIndex<Complex>((long)r * 3 + c).Should().Be(r == c ? C(1, 0) : C(0, 0));
        }

        [TestMethod]
        public void Identity_SByte()
        {
            var a = np.identity(5, typeof(sbyte));
            a.shape.Should().Equal(new long[] { 5, 5 });
            for (int r = 0; r < 5; r++)
                for (int c = 0; c < 5; c++)
                    a.GetAtIndex<sbyte>((long)r * 5 + c).Should().Be((sbyte)(r == c ? 1 : 0));
        }

        #endregion

        #region _like variants

        [TestMethod]
        public void ZerosLike_Half_PreservesDtype()
        {
            var p = np.zeros(new Shape(2, 3), typeof(Half));
            var a = np.zeros_like(p);
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 3 });
        }

        [TestMethod]
        public void ZerosLike_Complex_PreservesDtype()
        {
            var p = np.zeros(new Shape(2, 3), typeof(Complex));
            var a = np.zeros_like(p);
            a.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void ZerosLike_SByte_PreservesDtype()
        {
            var p = np.zeros(new Shape(2, 3), typeof(sbyte));
            var a = np.zeros_like(p);
            a.typecode.Should().Be(NPTypeCode.SByte);
        }

        [TestMethod]
        public void ZerosLike_DtypeOverride_Half()
        {
            // np.zeros_like(f64_arr, dtype=float16) → float16 zeros with same shape
            var p = np.zeros(new Shape(2, 3), typeof(double));
            var a = np.zeros_like(p, typeof(Half));
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 3 });
        }

        [TestMethod]
        public void ZerosLike_DtypeOverride_Complex()
        {
            var p = np.zeros(new Shape(2, 3), typeof(double));
            var a = np.zeros_like(p, typeof(Complex));
            a.typecode.Should().Be(NPTypeCode.Complex);
        }

        [TestMethod]
        public void OnesLike_DtypeOverride_SByte()
        {
            var p = np.zeros(new Shape(2, 3), typeof(double));
            var a = np.ones_like(p, typeof(sbyte));
            a.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 6; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void FullLike_Half()
        {
            var p = np.zeros(new Shape(2, 3), typeof(Half));
            var a = np.full_like(p, (Half)2.5);
            a.typecode.Should().Be(NPTypeCode.Half);
            ((double)a.GetAtIndex<Half>(0)).Should().BeApproximately(2.5, HalfTol);
        }

        [TestMethod]
        public void FullLike_Complex()
        {
            var p = np.zeros(new Shape(2, 3), typeof(Complex));
            var a = np.full_like(p, C(1, -1));
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, -1));
        }

        [TestMethod]
        public void FullLike_SByte()
        {
            var p = np.zeros(new Shape(2, 3), typeof(sbyte));
            var a = np.full_like(p, (sbyte)(-3));
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-3));
        }

        [TestMethod]
        public void EmptyLike_Half_ReturnsCorrectShapeAndDtype()
        {
            var p = np.zeros(new Shape(2, 3), typeof(Half));
            var a = np.empty_like(p);
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 3 });
            a.size.Should().Be(6);
        }

        #endregion

        #region meshgrid

        [TestMethod]
        public void Meshgrid_Half()
        {
            var x = np.array(new Half[] { (Half)1, (Half)2, (Half)3 });
            var y = np.array(new Half[] { (Half)10, (Half)20 });
            var tup = np.meshgrid(x, y);
            tup.Item1.typecode.Should().Be(NPTypeCode.Half);
            tup.Item1.shape.Should().Equal(new long[] { 2, 3 });
            tup.Item2.shape.Should().Equal(new long[] { 2, 3 });
            // Row 0: x values; Col 0: y values (xy indexing default)
            ((double)tup.Item1.GetAtIndex<Half>(0)).Should().Be(1);
            ((double)tup.Item1.GetAtIndex<Half>(5)).Should().Be(3);
            ((double)tup.Item2.GetAtIndex<Half>(0)).Should().Be(10);
            ((double)tup.Item2.GetAtIndex<Half>(5)).Should().Be(20);
        }

        [TestMethod]
        public void Meshgrid_Complex()
        {
            var x = np.array(new Complex[] { C(1, 0), C(2, 0) });
            var y = np.array(new Complex[] { C(0, 1), C(0, 2) });
            var tup = np.meshgrid(x, y);
            tup.Item1.typecode.Should().Be(NPTypeCode.Complex);
            tup.Item1.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
            tup.Item2.GetAtIndex<Complex>(0).Should().Be(C(0, 1));
        }

        [TestMethod]
        public void Meshgrid_SByte()
        {
            var x = np.array(new sbyte[] { 1, 2, 3 });
            var y = np.array(new sbyte[] { 10, 20 });
            var tup = np.meshgrid(x, y);
            tup.Item1.typecode.Should().Be(NPTypeCode.SByte);
            tup.Item1.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            tup.Item2.GetAtIndex<sbyte>(5).Should().Be((sbyte)20);
        }

        #endregion

        #region frombuffer

        [TestMethod]
        public void Frombuffer_Half()
        {
            var half = new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 };
            var bytes = new byte[half.Length * 2];
            unsafe
            {
                fixed (Half* p = half)
                fixed (byte* b = bytes)
                    Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            }
            var a = np.frombuffer(bytes, typeof(Half));
            a.size.Should().Be(4);
            a.typecode.Should().Be(NPTypeCode.Half);
            for (int i = 0; i < 4; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(i + 1.0);
        }

        [TestMethod]
        public void Frombuffer_Complex()
        {
            var cplx = new Complex[] { C(1, 0), C(2, 1), C(-3, 4) };
            var bytes = new byte[cplx.Length * 16];
            unsafe
            {
                fixed (Complex* p = cplx)
                fixed (byte* b = bytes)
                    Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            }
            var a = np.frombuffer(bytes, typeof(Complex));
            a.size.Should().Be(3);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
            a.GetAtIndex<Complex>(1).Should().Be(C(2, 1));
            a.GetAtIndex<Complex>(2).Should().Be(C(-3, 4));
        }

        [TestMethod]
        public void Frombuffer_SByte()
        {
            var sb = new sbyte[] { -128, -1, 0, 1, 127 };
            var bytes = new byte[sb.Length];
            unsafe
            {
                fixed (sbyte* p = sb)
                fixed (byte* b = bytes)
                    Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            }
            var a = np.frombuffer(bytes, typeof(sbyte));
            a.size.Should().Be(5);
            for (int i = 0; i < 5; i++)
                a.GetAtIndex<sbyte>(i).Should().Be(sb[i]);
        }

        [TestMethod]
        public void Frombuffer_Half_WithOffsetAndCount()
        {
            var half = new Half[] { (Half)1, (Half)2, (Half)3, (Half)4 };
            var bytes = new byte[half.Length * 2];
            unsafe
            {
                fixed (Half* p = half)
                fixed (byte* b = bytes)
                    Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            }
            var a = np.frombuffer(bytes, typeof(Half), count: 2, offset: 2);
            a.size.Should().Be(2);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(2.0);
            ((double)a.GetAtIndex<Half>(1)).Should().Be(3.0);
        }

        #endregion

        #region copy

        [TestMethod]
        public void Copy_Half_ReturnsIndependentBuffer()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Half).reshape(2, 3);
            var cp = np.copy(src);
            cp.typecode.Should().Be(NPTypeCode.Half);
            cp.shape.Should().Equal(new long[] { 2, 3 });
            for (int i = 0; i < 6; i++)
                ((double)cp.GetAtIndex<Half>(i)).Should().Be((double)i);
        }

        [TestMethod]
        public void Copy_Complex()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Complex).reshape(2, 3);
            var cp = np.copy(src);
            cp.typecode.Should().Be(NPTypeCode.Complex);
            for (int i = 0; i < 6; i++)
                cp.GetAtIndex<Complex>(i).Should().Be(C(i, 0));
        }

        [TestMethod]
        public void Copy_SByte()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.SByte).reshape(2, 3);
            var cp = np.copy(src);
            cp.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 6; i++)
                cp.GetAtIndex<sbyte>(i).Should().Be((sbyte)i);
        }

        #endregion

        #region asarray / asanyarray — B28 + B29 regression

        [TestMethod]
        public void B29_Asarray_NDArray_Half_DtypeOverride()
        {
            // np.asarray(float64_arr, dtype=float16) converts to float16
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Double).reshape(2, 3);
            var a = np.asarray(src, typeof(Half));
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 3 });
            for (int i = 0; i < 6; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be((double)i);
        }

        [TestMethod]
        public void B29_Asarray_NDArray_Complex_DtypeOverride()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Double).reshape(2, 3);
            var a = np.asarray(src, typeof(Complex));
            a.typecode.Should().Be(NPTypeCode.Complex);
            for (int i = 0; i < 6; i++)
                a.GetAtIndex<Complex>(i).Should().Be(C(i, 0));
        }

        [TestMethod]
        public void B29_Asarray_NDArray_SByte_DtypeOverride()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Double).reshape(2, 3);
            var a = np.asarray(src, typeof(sbyte));
            a.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 6; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)i);
        }

        [TestMethod]
        public void B29_Asarray_NDArray_SameDtype_ReturnsAsIs()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Half);
            var a = np.asarray(src, typeof(Half));
            // For same dtype we expect reference equality (no copy).
            ReferenceEquals(a, src).Should().BeTrue();
        }

        [TestMethod]
        public void B29_Asarray_NDArray_NullDtype_ReturnsAsIs()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Complex);
            var a = np.asarray(src, null);
            ReferenceEquals(a, src).Should().BeTrue();
        }

        [TestMethod]
        public void B28_Asanyarray_NDArray_Half_DtypeOverride()
        {
            // NumPy: np.asanyarray(f64_arr, dtype=float16) converts. NumSharp was ignoring dtype
            // on the NDArray fast-path.
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Double).reshape(2, 3);
            var a = np.asanyarray(src, typeof(Half));
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 3 });
            for (int i = 0; i < 6; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be((double)i);
        }

        [TestMethod]
        public void B28_Asanyarray_NDArray_Complex_DtypeOverride()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Double).reshape(2, 3);
            var a = np.asanyarray(src, typeof(Complex));
            a.typecode.Should().Be(NPTypeCode.Complex);
            for (int i = 0; i < 6; i++)
                a.GetAtIndex<Complex>(i).Should().Be(C(i, 0));
        }

        [TestMethod]
        public void B28_Asanyarray_NDArray_SByte_DtypeOverride()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Double).reshape(2, 3);
            var a = np.asanyarray(src, typeof(sbyte));
            a.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 6; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)i);
        }

        [TestMethod]
        public void B28_Asanyarray_NDArray_SameDtype_ReturnsAsIs()
        {
            var src = np.arange(0.0, 6.0, 1.0, NPTypeCode.Half);
            var a = np.asanyarray(src, typeof(Half));
            ReferenceEquals(a, src).Should().BeTrue();
        }

        #endregion

        #region B30 — frombuffer string dtype parser (Half/Complex/SByte codes + byte order)

        [TestMethod]
        public void B30_Frombuffer_StringDtype_f2_MapsToHalf()
        {
            // NumPy: np.frombuffer(bytes, dtype='f2') → float16 array
            var half = new Half[] { (Half)1, (Half)2, (Half)3 };
            var bytes = ToBytes(half);
            var a = np.frombuffer(bytes, "f2");
            a.typecode.Should().Be(NPTypeCode.Half);
            for (int i = 0; i < 3; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(i + 1.0);
        }

        [TestMethod]
        public void B30_Frombuffer_StringDtype_e_MapsToHalf()
        {
            // NumPy short code 'e' == float16
            var half = new Half[] { (Half)1, (Half)2, (Half)3 };
            var a = np.frombuffer(ToBytes(half), "e");
            a.typecode.Should().Be(NPTypeCode.Half);
        }

        [TestMethod]
        public void B30_Frombuffer_StringDtype_c16_MapsToComplex()
        {
            // NumPy: 'c16' == complex128
            var cplx = new Complex[] { C(1, 0), C(2, 1) };
            var a = np.frombuffer(ToBytes(cplx), "c16");
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
            a.GetAtIndex<Complex>(1).Should().Be(C(2, 1));
        }

        [TestMethod]
        public void B30_Frombuffer_StringDtype_D_MapsToComplex()
        {
            var cplx = new Complex[] { C(3, 4) };
            var a = np.frombuffer(ToBytes(cplx), "D");
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.GetAtIndex<Complex>(0).Should().Be(C(3, 4));
        }

        [TestMethod]
        public void B30_Frombuffer_StringDtype_i1_MapsToSByte_NotByte()
        {
            // Pre-fix 'i1' / 'b' incorrectly mapped to NPTypeCode.Byte (uint8)
            // NumPy: 'i1' == int8 → must return sbyte values
            var sb = new sbyte[] { -1, 0, 1 };
            var a = np.frombuffer(ToBytes(sb), "i1");
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-1));
            a.GetAtIndex<sbyte>(1).Should().Be((sbyte)0);
            a.GetAtIndex<sbyte>(2).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void B30_Frombuffer_StringDtype_b_MapsToSByte()
        {
            var sb = new sbyte[] { -128, 127 };
            var a = np.frombuffer(ToBytes(sb), "b");
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)(-128));
            a.GetAtIndex<sbyte>(1).Should().Be((sbyte)127);
        }

        #endregion

        #region B31 — ByteSwapInPlace covers Half and Complex

        [TestMethod]
        public void B31_Frombuffer_BigEndian_Half_SwapsCorrectly()
        {
            // Build little-endian representation, then byte-swap to simulate BE buffer.
            var half = new Half[] { (Half)1, (Half)2, (Half)3 };
            var bytes = ToBytes(half);
            var be = (byte[])bytes.Clone();
            for (int i = 0; i < be.Length; i += 2) (be[i], be[i + 1]) = (be[i + 1], be[i]);
            var a = np.frombuffer(be, ">f2");
            a.typecode.Should().Be(NPTypeCode.Half);
            for (int i = 0; i < 3; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(i + 1.0);
        }

        [TestMethod]
        public void B31_Frombuffer_BigEndian_Complex_SwapsCorrectly()
        {
            var cplx = new Complex[] { C(1, 0), C(2, 1) };
            var bytes = ToBytes(cplx);
            // Swap each 8-byte double independently (Complex = 2 doubles)
            var be = (byte[])bytes.Clone();
            for (int i = 0; i < be.Length; i += 8) Array.Reverse(be, i, 8);
            var a = np.frombuffer(be, ">c16");
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
            a.GetAtIndex<Complex>(1).Should().Be(C(2, 1));
        }

        #endregion

        #region B32 — np.eye rejects negative N and M

        [TestMethod]
        public void B32_Eye_NegativeN_ThrowsArgumentException()
        {
            Action act = () => np.eye(-1, dtype: typeof(Half));
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void B32_Eye_NegativeM_ThrowsArgumentException()
        {
            Action act = () => np.eye(3, -1, 0, typeof(Complex));
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void B32_Eye_ZeroNZeroM_ReturnsEmpty()
        {
            // 0×0 is valid — should return empty, not throw
            var a = np.eye(0, 0, 0, typeof(sbyte));
            a.size.Should().Be(0);
        }

        #endregion

        #region Round 12 — additional smoke tests from extended sweep

        [TestMethod]
        public void FullInference_Half_FromScalar()
        {
            // np.full(shape, half(2.5)) infers dtype from fill_value
            var a = np.full(new Shape(3), (Half)2.5);
            a.typecode.Should().Be(NPTypeCode.Half);
            ((double)a.GetAtIndex<Half>(0)).Should().BeApproximately(2.5, HalfTol);
        }

        [TestMethod]
        public void FullInference_Complex_FromScalar()
        {
            var a = np.full(new Shape(3), new Complex(1, 2));
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
        }

        [TestMethod]
        public void FullInference_SByte_FromScalar()
        {
            var a = np.full(new Shape(3), (sbyte)5);
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.GetAtIndex<sbyte>(0).Should().Be((sbyte)5);
        }

        [TestMethod]
        public void Arange_SByte_FloatStep_IntTruncation()
        {
            // NumPy arange(0,5,0.5,int8) computes delta_t = int8(0.5)=0 → all zeros
            var a = np.arange(0.0, 5.0, 0.5, NPTypeCode.SByte);
            a.size.Should().Be(10);
            for (int i = 0; i < 10; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)0);
        }

        [TestMethod]
        public void Eye_3x3_KExtremeDiagonal_Half()
        {
            // k = M-1 = 2: single element at (0,2)
            var a = np.eye(3, 3, 2, typeof(Half));
            var expected = new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < 9; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(expected[i]);
        }

        [TestMethod]
        public void Linspace_Half_N2_NoEndpoint()
        {
            // [start, start + (stop-start)/2] = [0, 2]
            var a = np.linspace(0.0, 4.0, 2L, false, NPTypeCode.Half);
            a.size.Should().Be(2);
            ((double)a.GetAtIndex<Half>(0)).Should().Be(0.0);
            ((double)a.GetAtIndex<Half>(1)).Should().Be(2.0);
        }

        [TestMethod]
        public void Zeros_4D_Half()
        {
            var a = np.zeros(new Shape(2, 2, 2, 2), typeof(Half));
            a.shape.Should().Equal(new long[] { 2, 2, 2, 2 });
            a.size.Should().Be(16);
            a.typecode.Should().Be(NPTypeCode.Half);
        }

        [TestMethod]
        public void Ones_5D_Complex()
        {
            var a = np.ones(new Shape(1, 2, 1, 2, 1), typeof(Complex));
            a.shape.Should().Equal(new long[] { 1, 2, 1, 2, 1 });
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
        }

        [TestMethod]
        public void Array3D_SByte()
        {
            var a = np.array(new sbyte[, ,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.shape.Should().Equal(new long[] { 2, 2, 2 });
            a.size.Should().Be(8);
            for (int i = 0; i < 8; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)(i + 1));
        }

        [TestMethod]
        public void MeshgridSparse_Half()
        {
            var x = np.array(new Half[] { (Half)1, (Half)2, (Half)3 });
            var y = np.array(new Half[] { (Half)10, (Half)20 });
            var kw = new Kwargs { indexing = "xy", sparse = true, copy = true };
            var tup = np.meshgrid(x, y, kw);
            tup.Item1.shape.Should().Equal(new long[] { 1, 3 });
            tup.Item2.shape.Should().Equal(new long[] { 2, 1 });
        }

        [TestMethod]
        public void MeshgridIJ_Complex()
        {
            var x = np.array(new Complex[] { C(1, 0), C(2, 0), C(3, 0) });
            var y = np.array(new Complex[] { C(0, 1), C(0, 2) });
            var kw = new Kwargs { indexing = "ij", sparse = false, copy = true };
            var tup = np.meshgrid(x, y, kw);
            // ij indexing: item1 shape (len(x), len(y)) = (3,2), item2 shape (3,2) too.
            tup.Item1.shape.Should().Equal(new long[] { 3, 2 });
            tup.Item2.shape.Should().Equal(new long[] { 3, 2 });
        }

        [TestMethod]
        public void ZerosLike_FromView_Half()
        {
            var baseArr = np.arange(0.0, 12.0, 1.0, NPTypeCode.Half).reshape(3, 4);
            var view = baseArr["0:2, 1:3"];
            var a = np.zeros_like(view);
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 2 });
            for (int i = 0; i < 4; i++)
                ((double)a.GetAtIndex<Half>(i)).Should().Be(0.0);
        }

        [TestMethod]
        public void OnesLike_FromStridedView_SByte()
        {
            var baseArr = np.arange(0.0, 12.0, 1.0, NPTypeCode.SByte).reshape(3, 4);
            var view = baseArr["::2"];  // rows 0 and 2 -> shape (2,4)
            var a = np.ones_like(view);
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.shape.Should().Equal(new long[] { 2, 4 });
            for (int i = 0; i < 8; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)1);
        }

        [TestMethod]
        public void ArangeLargeN_SByte_100Elements()
        {
            // NumPy wraps: arange(0,100,1,int8) → [0..99], no overflow since values fit in int8 up to 127
            var a = np.arange(0.0, 100.0, 1.0, NPTypeCode.SByte);
            a.size.Should().Be(100);
            for (int i = 0; i < 100; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)i);
        }

        [TestMethod]
        public void Zeros_AllZeroDimensions_ReturnsEmpty_Half()
        {
            var a = np.zeros(new Shape(0, 0, 0), typeof(Half));
            a.size.Should().Be(0);
            a.shape.Should().Equal(new long[] { 0, 0, 0 });
        }

        [TestMethod]
        public void Ones_ScalarShape_Complex()
        {
            var a = np.ones(Shape.NewScalar(), typeof(Complex));
            a.size.Should().Be(1);
            a.shape.Length.Should().Be(0);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 0));
        }

        [TestMethod]
        public void Frombuffer_Count0_Half_ReturnsEmpty()
        {
            var half = new Half[] { (Half)1, (Half)2 };
            var a = np.frombuffer(ToBytes(half), typeof(Half), count: 0);
            a.size.Should().Be(0);
            a.typecode.Should().Be(NPTypeCode.Half);
        }

        #endregion

        #region Helper

        private static byte[] ToBytes<T>(T[] arr) where T : unmanaged
        {
            var bytes = new byte[arr.Length * System.Runtime.CompilerServices.Unsafe.SizeOf<T>()];
            unsafe
            {
                fixed (T* p = arr)
                fixed (byte* b = bytes)
                    Buffer.MemoryCopy(p, b, bytes.Length, bytes.Length);
            }
            return bytes;
        }

        #endregion

        #region np.array — typed arrays for the 3 dtypes

        [TestMethod]
        public void Array_Half_1D()
        {
            var a = np.array(new Half[] { (Half)1, (Half)2, (Half)3 });
            a.typecode.Should().Be(NPTypeCode.Half);
            a.size.Should().Be(3);
        }

        [TestMethod]
        public void Array_Complex_1D()
        {
            var a = np.array(new Complex[] { C(1, 2), C(3, -4) });
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.GetAtIndex<Complex>(0).Should().Be(C(1, 2));
            a.GetAtIndex<Complex>(1).Should().Be(C(3, -4));
        }

        [TestMethod]
        public void Array_SByte_1D()
        {
            var a = np.array(new sbyte[] { 1, 2, 3 });
            a.typecode.Should().Be(NPTypeCode.SByte);
            for (int i = 0; i < 3; i++)
                a.GetAtIndex<sbyte>(i).Should().Be((sbyte)(i + 1));
        }

        [TestMethod]
        public void Array_Half_2D()
        {
            var a = np.array(new Half[,] { { (Half)1, (Half)2 }, { (Half)3, (Half)4 } });
            a.typecode.Should().Be(NPTypeCode.Half);
            a.shape.Should().Equal(new long[] { 2, 2 });
        }

        [TestMethod]
        public void Array_Complex_2D()
        {
            var a = np.array(new Complex[,] { { C(1, 0) }, { C(0, 1) } });
            a.typecode.Should().Be(NPTypeCode.Complex);
            a.shape.Should().Equal(new long[] { 2, 1 });
        }

        [TestMethod]
        public void Array_SByte_2D()
        {
            var a = np.array(new sbyte[,] { { 1, 2 }, { 3, 4 } });
            a.typecode.Should().Be(NPTypeCode.SByte);
            a.shape.Should().Equal(new long[] { 2, 2 });
        }

        #endregion
    }
}
