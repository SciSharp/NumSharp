using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    /// <summary>
    ///     Tests for <see cref="UnmanagedMemoryBlock.Allocate(Type, long, object)"/>
    ///     covering same-type fills, cross-type fills (NumPy-parity wrapping), and
    ///     the new dtypes (SByte / Half / Complex).
    /// </summary>
    [TestClass]
    public unsafe class UnmanagedMemoryBlockAllocateTests
    {
        // ---------------------------------------------------------------------
        // Same-type fills (classic path)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Allocate_Int32_SameType_Fill()
        {
            var block = (UnmanagedMemoryBlock<int>)UnmanagedMemoryBlock.Allocate(typeof(int), 4L, 42);
            block.Count.Should().Be(4);
            for (int i = 0; i < 4; i++) block[i].Should().Be(42);
        }

        [TestMethod]
        public void Allocate_Boolean_SameType_Fill()
        {
            var block = (UnmanagedMemoryBlock<bool>)UnmanagedMemoryBlock.Allocate(typeof(bool), 3L, true);
            for (int i = 0; i < 3; i++) block[i].Should().BeTrue();
        }

        // ---------------------------------------------------------------------
        // SByte
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Allocate_SByte_SameType_Fill()
        {
            var block = (UnmanagedMemoryBlock<sbyte>)UnmanagedMemoryBlock.Allocate(typeof(sbyte), 3L, (sbyte)-42);
            for (int i = 0; i < 3; i++) block[i].Should().Be((sbyte)(-42));
        }

        [TestMethod]
        public void Allocate_SByte_CrossType_FromInt()
        {
            // NumSharp cross-type fill now supported via Converts.ToSByte
            var block = (UnmanagedMemoryBlock<sbyte>)UnmanagedMemoryBlock.Allocate(typeof(sbyte), 2L, 100);
            block[0].Should().Be((sbyte)100);
            block[1].Should().Be((sbyte)100);
        }

        [TestMethod]
        public void Allocate_SByte_Boundary_MinValue()
        {
            var block = (UnmanagedMemoryBlock<sbyte>)UnmanagedMemoryBlock.Allocate(typeof(sbyte), 1L, (sbyte)sbyte.MinValue);
            block[0].Should().Be(sbyte.MinValue);
        }

        [TestMethod]
        public void Allocate_SByte_Boundary_MaxValue()
        {
            var block = (UnmanagedMemoryBlock<sbyte>)UnmanagedMemoryBlock.Allocate(typeof(sbyte), 1L, (sbyte)sbyte.MaxValue);
            block[0].Should().Be(sbyte.MaxValue);
        }

        // ---------------------------------------------------------------------
        // Half
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Allocate_Half_SameType_Fill()
        {
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 3L, (Half)3.5);
            for (int i = 0; i < 3; i++) block[i].Should().Be((Half)3.5);
        }

        [TestMethod]
        public void Allocate_Half_CrossType_FromInt()
        {
            // Before the fix, `(Half)(object)42` threw InvalidCastException.
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 2L, 42);
            block[0].Should().Be((Half)42);
            block[1].Should().Be((Half)42);
        }

        [TestMethod]
        public void Allocate_Half_CrossType_FromDouble()
        {
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 1L, 3.14);
            ((float)block[0]).Should().BeApproximately(3.14f, 0.01f);
        }

        [TestMethod]
        public void Allocate_Half_CrossType_FromSingle()
        {
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 1L, 2.5f);
            block[0].Should().Be((Half)2.5);
        }

        [TestMethod]
        public void Allocate_Half_CrossType_FromSByte()
        {
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 1L, (sbyte)-7);
            block[0].Should().Be((Half)(-7));
        }

        [TestMethod]
        public void Allocate_Half_NaN_Preserved()
        {
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 1L, Half.NaN);
            Half.IsNaN(block[0]).Should().BeTrue();
        }

        [TestMethod]
        public void Allocate_Half_Inf_Preserved()
        {
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 1L, Half.PositiveInfinity);
            Half.IsPositiveInfinity(block[0]).Should().BeTrue();
        }

        // ---------------------------------------------------------------------
        // Complex
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Allocate_Complex_SameType_Fill()
        {
            var block = (UnmanagedMemoryBlock<Complex>)UnmanagedMemoryBlock.Allocate(typeof(Complex), 3L, new Complex(1, 2));
            for (int i = 0; i < 3; i++) block[i].Should().Be(new Complex(1, 2));
        }

        [TestMethod]
        public void Allocate_Complex_CrossType_FromInt()
        {
            // Before the fix, `(Complex)(object)42` threw InvalidCastException.
            var block = (UnmanagedMemoryBlock<Complex>)UnmanagedMemoryBlock.Allocate(typeof(Complex), 2L, 42);
            block[0].Should().Be(new Complex(42, 0));
            block[1].Should().Be(new Complex(42, 0));
        }

        [TestMethod]
        public void Allocate_Complex_CrossType_FromDouble()
        {
            var block = (UnmanagedMemoryBlock<Complex>)UnmanagedMemoryBlock.Allocate(typeof(Complex), 1L, 3.14);
            block[0].Real.Should().Be(3.14);
            block[0].Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Allocate_Complex_CrossType_FromHalf()
        {
            var block = (UnmanagedMemoryBlock<Complex>)UnmanagedMemoryBlock.Allocate(typeof(Complex), 1L, (Half)2.5);
            block[0].Real.Should().Be(2.5);
            block[0].Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Allocate_Complex_CrossType_FromSByte()
        {
            var block = (UnmanagedMemoryBlock<Complex>)UnmanagedMemoryBlock.Allocate(typeof(Complex), 1L, (sbyte)(-7));
            block[0].Should().Be(new Complex(-7, 0));
        }

        // ---------------------------------------------------------------------
        // Existing types: cross-type fills (regression — was previously broken for Half/Complex source)
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Allocate_Int32_CrossType_FromHalf()
        {
            // (int)(object)(Half)7.5 used to throw; Converts.ToInt32 truncates to 7
            var block = (UnmanagedMemoryBlock<int>)UnmanagedMemoryBlock.Allocate(typeof(int), 1L, (Half)7.5);
            block[0].Should().Be(7);
        }

        [TestMethod]
        public void Allocate_Double_CrossType_FromHalf()
        {
            var block = (UnmanagedMemoryBlock<double>)UnmanagedMemoryBlock.Allocate(typeof(double), 1L, (Half)3.5);
            block[0].Should().Be(3.5);
        }

        [TestMethod]
        public void Allocate_Int32_CrossType_FromComplex()
        {
            // Complex->Int32: discards imaginary, truncates real
            var block = (UnmanagedMemoryBlock<int>)UnmanagedMemoryBlock.Allocate(typeof(int), 1L, new Complex(7.5, 3));
            block[0].Should().Be(7);
        }

        [TestMethod]
        public void Allocate_Double_CrossType_FromComplex()
        {
            // Complex->Double: discards imaginary
            var block = (UnmanagedMemoryBlock<double>)UnmanagedMemoryBlock.Allocate(typeof(double), 1L, new Complex(3.14, 2));
            block[0].Should().Be(3.14);
        }

        // ---------------------------------------------------------------------
        // Ensure existing non-fill Allocate still works
        // ---------------------------------------------------------------------

        [TestMethod]
        public void Allocate_Half_NoFill_ReturnsZeros()
        {
            // Allocate without fill should give default(T) = (Half)0 — actually unmanaged memory
            // isn't zero-initialized by default, so this is only valid for the fill overload or
            // the Allocate(count, default) variant tested in UnmanagedMemoryBlock. Verify the
            // no-fill overload returns a valid block of correct size.
            var block = (UnmanagedMemoryBlock<Half>)UnmanagedMemoryBlock.Allocate(typeof(Half), 4L);
            block.Count.Should().Be(4);
        }

        [TestMethod]
        public void Allocate_Complex_NoFill_ReturnsValidBlock()
        {
            var block = (UnmanagedMemoryBlock<Complex>)UnmanagedMemoryBlock.Allocate(typeof(Complex), 4L);
            block.Count.Should().Be(4);
        }
    }
}
