using System;
using System.Buffers;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends.Unmanaged;
using NumSharp.Interop.OnnxRuntime;
// NumSharp.Core has a memory manager of the same name; the one under test is the bridge's.
using IntManager = NumSharp.Interop.OnnxRuntime.UnmanagedMemoryManager<int>;

namespace NumSharp.Tests.Interop.OnnxRuntime
{
    /// <summary>
    ///     The guards of <see cref="NumSharp.Interop.OnnxRuntime.UnmanagedMemoryManager{T}"/>, the object behind the <see cref="Memory{T}"/> of an
    ///     <c>AsDenseTensor</c> handle, exercised directly. A caller of the bridge reaches the manager only through a
    ///     <see cref="Memory{T}"/> whose index is always in range, so these paths had no test; they are what stands
    ///     between a wrong internal call and a read outside a NumSharp buffer.
    /// </summary>
    [TestClass]
    public unsafe class MemoryManagerTests : OnnxTestBase
    {
        /// <summary>
        ///     The constructor refuses a negative length and a null buffer. A pin addresses the element asked for,
        ///     holds its own reference on the buffer until it is disposed, and an index past the end is refused (the end
        ///     itself is a legal address, as for any <see cref="Memory{T}"/>). A pin token cannot be pinned again.
        /// </summary>
        [TestMethod]
        public void Manager_ChecksItsArguments_AndAPinHoldsOneReference()
        {
            using NDArray nd = Arange(NPTypeCode.Int32, 4);
            IArraySlice buffer = nd.Storage.InternalArray;
            int* first = (int*)buffer.Address;

            new Action(() => new IntManager(first, -1, buffer, writeable: true)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => new IntManager(first, 4, null, writeable: true)).Should().Throw<ArgumentNullException>();

            var manager = new IntManager(first, 4, buffer, writeable: false);
            manager.Length.Should().Be(4);
            manager.Writeable.Should().BeFalse();
            manager.IsDisposed.Should().BeFalse();
            manager.GetSpan().ToArray().Should().Equal(0, 1, 2, 3);
            buffer.IsUniquelyReferenced.Should().BeTrue("the manager takes no reference of its own");

            MemoryHandle pin = manager.Pin(2);
            ((IntPtr)pin.Pointer).Should().Be((IntPtr)(first + 2), "the pin addresses the element asked for");
            buffer.IsUniquelyReferenced.Should().BeFalse("a pin holds a reference on the buffer");
            pin.Dispose();
            buffer.IsUniquelyReferenced.Should().BeTrue("and gives it back when it is disposed");
            pin.Dispose();   // a second dispose of the same pin gives nothing back
            buffer.IsUniquelyReferenced.Should().BeTrue();

            using (MemoryHandle end = manager.Pin(4))
                ((IntPtr)end.Pointer).Should().Be((IntPtr)(first + 4), "one past the last element is a legal address");
            new Action(() => manager.Pin(5)).Should().Throw<ArgumentOutOfRangeException>();
            new Action(() => manager.Pin(-1)).Should().Throw<ArgumentOutOfRangeException>();
            buffer.IsUniquelyReferenced.Should().BeTrue("a refused pin takes no reference");

            manager.Unpin();   // does nothing: every pin carries its own token

            var token = new PinToken(buffer);
            buffer.TryAddRef().Should().BeTrue();   // the reference the token takes over
            token.IsReleased.Should().BeFalse();
            new Action(() => token.Pin(0)).Should().Throw<NotSupportedException>().WithMessage("*already one pin*");
            token.Unpin();
            token.IsReleased.Should().BeTrue();
            buffer.IsUniquelyReferenced.Should().BeTrue("the token gave its reference back, once");
        }

        /// <summary>
        ///     A manager whose handle has been disposed refuses new spans and new pins, and a pin of memory whose buffer
        ///     has been released is refused instead of addressing memory the pool may hand to another array.
        /// </summary>
        [TestMethod]
        public void Manager_RefusesNewAccess_OnceDisposed_OrOnceItsBufferIsReleased()
        {
            NDArray nd = Arange(NPTypeCode.Int32, 4);
            IArraySlice buffer = nd.Storage.InternalArray;
            var manager = new IntManager((int*)buffer.Address, 4, buffer, writeable: true);

            // The buffer goes first (the array is disposed while the manager still fronts it): a pin must not succeed.
            nd.Dispose();
            new Action(() => manager.Pin()).Should().Throw<ObjectDisposedException>().WithMessage("*already been released*");

            ((IDisposable)manager).Dispose();
            manager.IsDisposed.Should().BeTrue();
            new Action(() => manager.GetSpan()).Should().Throw<ObjectDisposedException>().WithMessage("*OrtTensor handle was disposed*");
            new Action(() => manager.Pin()).Should().Throw<ObjectDisposedException>().WithMessage("*OrtTensor handle was disposed*");
        }
    }
}
