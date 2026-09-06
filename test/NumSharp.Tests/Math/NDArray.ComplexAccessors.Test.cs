using System;
using System.Numerics;

namespace NumSharp.Tests.Math
{
    /// <summary>
    ///     ndarray.real / ndarray.imag (properties, get + set) and ndarray.conj / ndarray.conjugate
    ///     (methods). All behavior probed against NumPy 2.4.2.
    /// </summary>
    [TestClass]
    public class NDArray_ComplexAccessors_Test
    {
        // ---- real / imag getters ---------------------------------------------------------

        [TestMethod]
        public void Real_Complex_IsWriteableViewSharingMemory()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            var r = c.real;
            r.dtype.Should().Be(typeof(double));
            r.Shape.IsWriteable.Should().BeTrue();
            r.GetDouble(0).Should().Be(1); r.GetDouble(1).Should().Be(3);
            r[0] = (NDArray)9.0;   // write-through
            ((Complex)c.GetAtIndex(0)).Should().Be(new Complex(9, 2));
        }

        [TestMethod]
        public void Imag_Complex_IsWriteableViewSharingMemory()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            var im = c.imag;
            im.GetDouble(0).Should().Be(2); im.GetDouble(1).Should().Be(4);
            im[1] = (NDArray)7.0;
            ((Complex)c.GetAtIndex(1)).Should().Be(new Complex(3, 7));
        }

        [TestMethod]
        public void Real_RealArray_ReturnsSelf()
        {
            var f = np.array(new[] { 1.0, 2.0, 3.0 });
            ReferenceEquals(f.real, f).Should().BeTrue();
        }

        [TestMethod]
        public void Imag_RealArray_IsReadOnlyZeros()
        {
            var f = np.array(new[] { 1.0, 2.0, 3.0 });
            var im = f.imag;
            im.Shape.IsWriteable.Should().BeFalse();
            im.dtype.Should().Be(typeof(double));
            for (int i = 0; i < 3; i++) im.GetDouble(i).Should().Be(0);
        }

        [TestMethod]
        public void Real_Imag_IntArray_PreservesDtype()
        {
            var i = np.array(new[] { 1, 2, 3 });
            i.real.dtype.Should().Be(typeof(int));
            i.imag.dtype.Should().Be(typeof(int));
            for (int k = 0; k < 3; k++) i.imag.GetInt32(k).Should().Be(0);
        }

        // ---- real setter -----------------------------------------------------------------

        [TestMethod]
        public void RealSet_RealArray_OverwritesWholeArray()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            a.real = (NDArray)5;
            for (int i = 0; i < 3; i++) a.GetDouble(i).Should().Be(5);
        }

        [TestMethod]
        public void RealSet_Complex_OverwritesRealLaneOnly()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            c.real = (NDArray)5;
            ((Complex)c.GetAtIndex(0)).Should().Be(new Complex(5, 2));
            ((Complex)c.GetAtIndex(1)).Should().Be(new Complex(5, 4));
        }

        [TestMethod]
        public void RealSet_ElementWiseAndBroadcast()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            c.real = np.array(new[] { 10, 20 });
            ((Complex)c.GetAtIndex(0)).Should().Be(new Complex(10, 2));
            ((Complex)c.GetAtIndex(1)).Should().Be(new Complex(20, 4));

            var m = np.zeros(new Shape(2, 3));
            m.real = np.array(new[] { 1.0, 2.0, 3.0 });   // broadcast row
            m.GetDouble(0, 0).Should().Be(1); m.GetDouble(1, 2).Should().Be(3);
        }

        [TestMethod]
        public void RealSet_UnsafeCast_TruncateAndWrap()
        {
            var i32 = np.array(new[] { 1, 2, 3 });
            i32.real = (NDArray)3.9;          // float -> int truncates toward zero
            for (int i = 0; i < 3; i++) i32.GetInt32(i).Should().Be(3);

            var i8 = np.array(new sbyte[] { 1, 2, 3 });
            i8.real = (NDArray)300;           // out-of-range int wraps (300 -> 44)
            for (int i = 0; i < 3; i++) i8.GetSByte(i).Should().Be(44);
        }

        [TestMethod]
        public void RealSet_ComplexSource_KeepsRealPart()
        {
            var c = np.array(new[] { new Complex(1, 2) });
            c.real = (NDArray)new Complex(5, 9);   // complex -> float takes real part
            ((Complex)c.GetAtIndex(0)).Should().Be(new Complex(5, 2));
        }

        [TestMethod]
        public void RealSet_Strided_WritesThrough()
        {
            var cs = np.array(new[] { new Complex(1, 1), new Complex(2, 2), new Complex(3, 3), new Complex(4, 4) })["::2"];
            cs.real = (NDArray)9;
            ((Complex)cs.GetAtIndex(0)).Should().Be(new Complex(9, 1));
            ((Complex)cs.GetAtIndex(1)).Should().Be(new Complex(9, 3));
        }

        [TestMethod]
        public void RealSet_ReadOnly_Raises()
        {
            var b = np.broadcast_to(np.array(new[] { 1.0 }), new Shape(3));
            ((Action)(() => b.real = (NDArray)5)).Should().Throw<NumSharpException>().WithMessage("*read-only*");
        }

        // ---- imag setter -----------------------------------------------------------------

        [TestMethod]
        public void ImagSet_Complex_OverwritesImagLane()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            c.imag = (NDArray)3;
            ((Complex)c.GetAtIndex(0)).Should().Be(new Complex(1, 3));
            ((Complex)c.GetAtIndex(1)).Should().Be(new Complex(3, 3));
        }

        [TestMethod]
        public void ImagSet_RealArray_RaisesTypeError()
        {
            var a = np.array(new[] { 1.0, 2.0, 3.0 });
            ((Action)(() => a.imag = (NDArray)3)).Should().Throw<TypeError>()
                .WithMessage("*does not have imaginary part to set*");
        }

        // ---- conj / conjugate ------------------------------------------------------------

        [TestMethod]
        public void Conjugate_Complex_FlipsImag()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, -4) });
            ((Complex)c.conj().GetAtIndex(0)).Should().Be(new Complex(1, -2));
            ((Complex)c.conjugate().GetAtIndex(1)).Should().Be(new Complex(3, 4));
        }

        [TestMethod]
        public void Conjugate_RealArray_ReturnsSelf()
        {
            var f = np.array(new[] { 1.0, 2.0 });
            ReferenceEquals(f.conj(), f).Should().BeTrue();
            ReferenceEquals(f.conjugate(), f).Should().BeTrue();
        }

        [TestMethod]
        public void Conjugate_Bool_PreservesBool_MethodNotFunction()
        {
            // The METHOD returns self (bool preserved); the FUNCTION np.conj promotes bool -> int8.
            var b = np.array(new[] { true, false });
            b.conj().dtype.Should().Be(typeof(bool));
            b.conjugate().dtype.Should().Be(typeof(bool));
            np.conj(b).dtype.Should().Be(typeof(sbyte));   // contrast: function promotes
        }

        [TestMethod]
        public void Conjugate_Int_PreservesDtype()
            => np.array(new[] { 1, 2 }).conjugate().dtype.Should().Be(typeof(int));

        [TestMethod]
        public void Conjugate_Complex_OutParam()
        {
            var c = np.array(new[] { new Complex(1, 2), new Complex(3, 4) });
            var o = np.zeros(new Shape(2), NPTypeCode.Complex);
            var r = c.conjugate(o);
            ReferenceEquals(r, o).Should().BeTrue();
            ((Complex)o.GetAtIndex(0)).Should().Be(new Complex(1, -2));
            ((Complex)o.GetAtIndex(1)).Should().Be(new Complex(3, -4));
        }

        [TestMethod]
        public void Conjugate_Real_OutParam_CopiesSelf()
        {
            var a = np.array(new[] { 1.0, 2.0 });
            var o = np.zeros(new Shape(2));
            var r = a.conjugate(o);
            ReferenceEquals(r, o).Should().BeTrue();
            o.GetDouble(0).Should().Be(1); o.GetDouble(1).Should().Be(2);
        }
    }
}
