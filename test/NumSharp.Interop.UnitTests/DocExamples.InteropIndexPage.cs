using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Executable proof for <c>docs/website-src/docs/interop/index.md</c> — the page that states the
    ///     contract every NumSharp bridge builds on.
    ///
    ///     <para>Unlike the other doc-example classes this one needs no Python: the page documents the
    ///     three <b>NumSharp</b> capabilities a bridge is made of (raw layout access, external-memory
    ///     wrapping, atomic reference counting), so it runs even on machines without an interpreter.</para>
    /// </summary>
    [TestClass]
    public class DocExamples_InteropIndexPage
    {
        // ============================  ## The Interop Contract  ==================================

        /// <summary>
        ///     Capability 1 — "an <c>NDArray</c> exposes its base address, element strides, offset and
        ///     dtype, so any strided-array convention can be expressed without copying".
        /// </summary>
        [TestMethod]
        public unsafe void Contract_RawLayoutAccess_ExposesAddressStridesOffsetAndDtype()
        {
            var nd = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);
            var window = nd["1:3, ::2"];

            ((IntPtr)window.Storage.Address).Should().NotBe(IntPtr.Zero, "base address");
            window.Shape.Strides.Should().Equal(new long[] { 6, 2 }, "element strides, not byte strides");
            window.Shape.Offset.Should().Be(6, "offset into the shared buffer");
            window.typecode.Should().Be(NPTypeCode.Double, "dtype");

            // "without copying": the window addresses the same buffer as its base.
            ((IntPtr)window.Storage.Address).Should().Be((IntPtr)nd.Storage.Address,
                "a view shares the base address; the offset does the addressing");
            *((double*)window.Storage.Address + window.Shape.Offset) = -1.0;
            nd.GetDouble(1, 0).Should().Be(-1.0, "the strided window really is the same memory");
        }

        /// <summary>
        ///     Capability 2 — the code block on the page, run verbatim: "The primitive every bridge is
        ///     made of: wrap foreign memory with a release hook."
        /// </summary>
        [TestMethod]
        public unsafe void Contract_ExternalMemoryWrapping_TheDocumentedPrimitive()
        {
            int rows = 2, cols = 3;
            long length = rows * cols;
            byte* ptr = (byte*)NativeMemory.Alloc((nuint)length);
            for (int i = 0; i < length; i++) ptr[i] = (byte)(i + 1);

            bool released = false;
            Action onLastReferenceReleased = () => released = true;

            // The primitive every bridge is made of: wrap foreign memory with a release hook.
            var nd = new NDArray(new UnmanagedStorage(
                new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, length, onLastReferenceReleased)),
                new Shape(rows, cols)));

            nd.shape.Should().Equal(new long[] { rows, cols });
            nd.typecode.Should().Be(NPTypeCode.Byte);
            nd.GetByte(1, 2).Should().Be(6, "foreign memory is read in place, not copied");

            // NumSharp kernels run over it directly — the page's claim for this capability.
            ((int)np.sum(nd)).Should().Be(21);

            // Writing through NumSharp writes the foreign buffer.
            nd[0, 0] = (NDArray)(byte)200;
            ptr[0].Should().Be(200);

            released.Should().BeFalse("the hook must not fire while a reference lives");
            nd.Dispose();
            released.Should().BeTrue("the hook fires when the last reference is released");

            NativeMemory.Free(ptr);
        }

        /// <summary>
        ///     Capability 3 — "The release hook fires when the <b>last</b> NumSharp reference to that
        ///     block — the original array <i>or any view derived from it</i> — goes away."
        /// </summary>
        [TestMethod]
        public unsafe void Contract_ReleaseHook_FiresOnTheLastReference_IncludingDerivedViews()
        {
            byte* ptr = (byte*)NativeMemory.Alloc(8);
            for (int i = 0; i < 8; i++) ptr[i] = (byte)i;

            bool released = false;
            var nd = new NDArray(new UnmanagedStorage(
                new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, 8, () => released = true)),
                new Shape(8)));

            NDArray derived = nd["2:"];        // a view derived from the same block

            nd.Dispose();                      // dropping the ORIGINAL is not enough...
            released.Should().BeFalse("a derived view still references the block");
            derived.GetByte(0).Should().Be(2, "and it is still valid to read");

            derived.Dispose();                 // ...the LAST reference is what fires the hook
            released.Should().BeTrue("the refcount decides, not disposal order");

            NativeMemory.Free(ptr);
        }

        /// <summary>
        ///     "...whether by <c>Dispose()</c> or by the GC" — the other half of the same sentence.
        /// </summary>
        [TestMethod]
        public unsafe void Contract_ReleaseHook_AlsoFiresByGarbageCollection()
        {
            byte* ptr = (byte*)NativeMemory.Alloc(8);
            var flag = new bool[1];

            WrapAndAbandon(ptr, flag);         // NoInlining: a debug-build frame keeps temps alive

            for (int i = 0; i < 20 && !flag[0]; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }

            flag[0].Should().BeTrue("the finalizer safety net releases the block even without Dispose()");
            NativeMemory.Free(ptr);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static unsafe void WrapAndAbandon(byte* ptr, bool[] flag)
        {
            var nd = new NDArray(new UnmanagedStorage(
                new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, 8, () => flag[0] = true)),
                new Shape(8)));
            nd.GetByte(0);                     // touch it so the wrap is not optimized away
        }

        /// <summary>
        ///     "Guards like <c>ndarray.resize(refcheck: true)</c> see those references too, mirroring
        ///     NumPy's own protection for exported buffers." The reference an extra NumSharp view holds
        ///     is the same ARC reference a bridge takes, so the guard can be shown without Python;
        ///     <see cref="LifetimeTests.NumSharpResize_IsBlockedByTheExportPin"/> shows it for a real
        ///     Python-side export.
        /// </summary>
        [TestMethod]
        public void Contract_RefcheckGuard_SeesOtherReferencesToTheBlock()
        {
            var nd = np.arange(8).astype(NPTypeCode.Double);
            NDArray secondReference = nd["2:"];

            ((Action)(() => nd.resize(new Shape(16))))
                .Should().Throw<IncorrectShapeException>()
                .WithMessage("*cannot resize an array that references or is referenced*",
                    "NumPy's own wording — the refcount guard sees the second reference");

            secondReference.Dispose();
            nd.resize(new Shape(16));   // ...and lifting the reference lifts the guard
            nd.size.Should().Be(16);
        }

        /// <summary>
        ///     The caveat the page spells out under the primitive: the bare wrap <b>claims ownership</b>,
        ///     so a size-changing <c>resize</c> silently reallocates AWAY from the foreign pointer. A
        ///     bridge that must stay attached has to <c>Alias</c> the storage — which is exactly what
        ///     <c>NDArrayPythonInterop</c>'s import path does, and why an imported view reports numpy's
        ///     <c>owndata == False</c> semantics.
        /// </summary>
        [TestMethod]
        public unsafe void Contract_BareWrapClaimsOwnership_AliasIsWhatKeepsItAttached()
        {
            // --- the bare wrap: NumSharp believes it owns the block, so resize reallocates.
            byte* ptr = (byte*)NativeMemory.Alloc(8);
            bool released = false;
            var owning = new NDArray(new UnmanagedStorage(
                new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, 8, () => released = true)),
                new Shape(8)));

            ((IntPtr)owning.Storage.Address).Should().Be((IntPtr)ptr);
            owning.resize(new Shape(16));   // succeeds — and detaches
            ((IntPtr)owning.Storage.Address).Should().NotBe((IntPtr)ptr,
                "the bare wrap claims ownership, so a growing resize moves to fresh NumSharp memory");
            released.Should().BeTrue("the release hook fired as the foreign block was let go");
            owning.Dispose();
            NativeMemory.Free(ptr);

            // --- the Alias form the pythonnet bridge actually uses: view semantics, stays attached.
            byte* p2 = (byte*)NativeMemory.Alloc(8);
            var aliased = new NDArray(
                new UnmanagedStorage(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(p2, 8, () => { })),
                                     Shape.Vector(8))
                    .Alias(new Shape(8)));

            ((Action)(() => aliased.resize(new Shape(16))))
                .Should().Throw<IncorrectShapeException>()
                .WithMessage("*cannot resize this array: it does not own its data*",
                    "aliasing gives the array numpy's owndata == False semantics, so it cannot detach");

            aliased.Dispose();
            NativeMemory.Free(p2);
        }

        // ============================  ## Official Bridges  ======================================

        /// <summary>
        ///     The bridges table names <c>NumSharp.Interop.pythonnet</c> as the reference
        ///     implementation of the contract above. This pins that the assembly is really present and
        ///     exposes the four verbs the linked page documents.
        /// </summary>
        [TestMethod]
        public void OfficialBridges_ThePythonnetRow_NamesARealAssembly()
        {
            var t = typeof(NumSharp.Interop.PythonNet.NDArrayPythonInterop);
            t.Assembly.GetName().Name.Should().Be("NumSharp.Interop.pythonnet");

            foreach (string verb in new[] { "ToNumpy", "ToNumpyCopy", "ToNDArray", "ToNDArrayView" })
                t.GetMethods().Should().Contain(m => m.Name == verb, $"the four verbs include {verb}");
        }
    }
}
