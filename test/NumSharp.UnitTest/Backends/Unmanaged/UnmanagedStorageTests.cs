using System;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.UnitTest.Backends.Unmanaged
{
    [TestClass]
    public class UnmanagedStorageTests
    {
        [TestMethod]
        public unsafe void CopyTo()
        {
            var src = np.arange(10).astype(NPTypeCode.Int32).Storage;
            var dst = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(10, -1));
            src.CopyTo(dst.Address);

            for (int i = 0; i < dst.Count; i++)
                dst[i].Should().Be(i);
        }

        [TestMethod]
        public unsafe void CopyTo_Sliced()
        {
            var src = np.arange(20).astype(NPTypeCode.Int32)["0:10"].Storage;
            var dst = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(10, -1));
            src.CopyTo(dst.Address);

            for (int i = 0; i < dst.Count; i++)
                dst[i].Should().Be(i);
        }

        [TestMethod]
        public unsafe void CopyTo_Corruption()
        {
            var np1 = np.arange(1,7).reshape(3, 2).astype(NPTypeCode.Double);
            var mean = np.mean(np1, keepdims: true);

            var src = np.arange(20).astype(NPTypeCode.Int32)["0:10"].Storage;
            var dst = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(10, -1));
            src.CopyTo(dst.Address);

            for (int i = 0; i < dst.Count; i++)
                dst[i].Should().Be(i);
        }

        [TestMethod]
        public unsafe void CopyTo_Block()
        {
            var src = np.arange(10).astype(NPTypeCode.Int32).Storage;
            var dst = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(10, -1));
            src.CopyTo(dst);

            for (int i = 0; i < dst.Count; i++)
                dst[i].Should().Be(i);
        }

        [TestMethod]
        public unsafe void CopyTo_Block_Sliced()
        {
            var src = np.arange(20).astype(NPTypeCode.Int32)["0:10"].Storage;
            var dst = new ArraySlice<int>(new UnmanagedMemoryBlock<int>(10, -1));
            src.CopyTo(dst);

            for (int i = 0; i < dst.Count; i++)
                dst[i].Should().Be(i);
        }

        [TestMethod]
        public unsafe void CopyTo_Array()
        {
            var src = np.arange(10).astype(NPTypeCode.Int32).Storage;
            var dst = new int[10];
            src.CopyTo(dst);

            for (int i = 0; i < dst.Length; i++)
                dst[i].Should().Be(i);
        }

        [TestMethod]
        public unsafe void CopyTo_Sliced_Array()
        {
            var src = np.arange(20).astype(NPTypeCode.Int32)["0:10"].Storage;
            var dst = new int[10];
            src.CopyTo(dst);

            for (int i = 0; i < dst.Length; i++)
                dst[i].Should().Be(i);
        }
    }
}
