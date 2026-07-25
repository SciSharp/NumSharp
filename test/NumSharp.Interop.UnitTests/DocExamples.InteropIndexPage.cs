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
    ///     Executable proof for <c>docs/website-src/docs/interop/index.md</c> — the landing page that
    ///     states the contract every NumSharp bridge builds on.
    ///
    ///     <para>This class needs no Python: the page's contract is three <b>NumSharp</b> capabilities
    ///     (raw layout access, external-memory wrapping, last-reference release), so its gates run
    ///     even on machines without an interpreter. Tests read no markdown and assert no prose
    ///     (the law of <c>98e6045a</c>).</para>
    /// </summary>
    [TestClass]
    public class DocExamples_InteropIndexPage
    {
        // ============================  ## The contract  ==========================================

        /// <summary>
        ///     "A layout is four numbers, and NumSharp exposes all four" — the page's window block
        ///     and its transcript: shared base address, element strides `[6, 2]`, offset `6`,
        ///     dtype `Double`.
        /// </summary>
        [TestMethod]
        public unsafe void Contract_RawLayoutAccess_ExposesAddressStridesOffsetAndDtype()
        {
            var nd = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);
            var window = nd["1:3, ::2"];

            ((IntPtr)window.Storage.Address).Should().Be((IntPtr)nd.Storage.Address,
                "views share the base pointer; the offset does the addressing");
            window.Shape.Strides.Should().Equal(new long[] { 6, 2 }, "element strides, not byte strides");
            window.Shape.Offset.Should().Be(6);
            window.typecode.Should().Be(NPTypeCode.Double);

            // "No elements move": addressing through the four numbers reaches the same memory.
            *((double*)window.Storage.Address + window.Shape.Offset) = -1.0;
            nd.GetDouble(1, 0).Should().Be(-1.0, "the strided window really is the same memory");
        }

        /// <summary>
        ///     "One primitive: wrap a pointer with a release hook" — the page's block, run
        ///     verbatim, with its transcript: reads in place, kernels over it, writes land, the hook
        ///     fires on Dispose.
        /// </summary>
        [TestMethod]
        public unsafe void Contract_ExternalMemoryWrapping_TheDocumentedPrimitive()
        {
            byte* ptr = (byte*)NativeMemory.Alloc(6);            // any foreign allocation
            for (int i = 0; i < 6; i++) ptr[i] = (byte)(i + 1);

            bool released = false;
            Action onLastReferenceReleased = () => released = true;

            var nd = new NDArray(new UnmanagedStorage(
                new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, 6, onLastReferenceReleased)),
                new Shape(2, 3)));

            nd.GetByte(1, 2).Should().Be(6, "nd.GetByte(1, 2) reads ptr[5] — in place, not copied");
            ((int)np.sum(nd)).Should().Be(21, "NumSharp kernels run over the foreign buffer");

            nd[0, 0] = (NDArray)(byte)200;
            ptr[0].Should().Be(200, "writes land in the foreign memory");

            released.Should().BeFalse("the hook must not fire while a reference lives");
            nd.Dispose();
            released.Should().BeTrue("the hook fires when NumSharp is done with the memory");

            NativeMemory.Free(ptr);
        }

        /// <summary>
        ///     "The hook fires on the last reference to the memory block — original or derived view" —
        ///     disposing the original frees nothing while a slice lives; the refcount decides, not
        ///     disposal order.
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
        ///     "the GC finalizer is the safety net when nothing was disposed at all" — the transcript's
        ///     third line.
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
        ///     "The same references feed NumSharp's resize guard … `cannot resize an array that
        ///     references or is referenced by another array in this way`" — the page's verbatim
        ///     NumPy-worded refusal, and that lifting the reference lifts the guard.
        /// </summary>
        [TestMethod]
        public void Contract_RefcheckGuard_SeesOtherReferencesToTheBlock()
        {
            var nd = np.arange(8).astype(NPTypeCode.Double);
            NDArray secondReference = nd["2:"];

            // The message reproduces NumPy's own, INCLUDING its line wrap after "referenced" —
            // hence two fragments rather than one sentence-spanning wildcard.
            ((Action)(() => nd.resize(new Shape(16))))
                .Should().Throw<IncorrectShapeException>()
                .WithMessage("*cannot resize an array that references or is referenced*")
                .WithMessage("*by another array in this way*");

            secondReference.Dispose();
            nd.resize(new Shape(16));   // ...and lifting the reference lifts the guard
            nd.size.Should().Be(16);
        }

        /// <summary>
        ///     "A bare wrap claims ownership: a growing `resize` succeeds by reallocating … A bridge
        ///     that must stay attached aliases the storage instead" — both transcript halves: the
        ///     address change + fired hook, and the aliased refusal.
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
                "a growing resize moves to fresh NumSharp memory, silently detaching from the foreign pointer");
            released.Should().BeTrue("the release hook fired as the foreign block was let go");
            owning.Dispose();
            NativeMemory.Free(ptr);

            // --- the Alias form the pythonnet bridge actually uses: view semantics, stays attached.
            byte* p = (byte*)NativeMemory.Alloc(8);
            var attached = new NDArray(
                new UnmanagedStorage(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(p, 8, () => { })),
                                     Shape.Vector(8))
                    .Alias(new Shape(8)));

            ((Action)(() => attached.resize(new Shape(16))))
                .Should().Throw<IncorrectShapeException>()
                .WithMessage("*cannot resize this array: it does not own its data*",
                    "aliasing gives the array numpy's owndata == False semantics, so it cannot detach");

            attached.Dispose();
            NativeMemory.Free(p);
        }

        // ============================  ## The bridges  ===========================================

        /// <summary>
        ///     The bridge table's first two rows name <c>NumSharp.Interop.pythonnet</c>. This pins that
        ///     the assembly is real and ships the four verbs the linked pages document.
        /// </summary>
        [TestMethod]
        public void Bridges_ThePythonnetPackage_ShipsTheFourVerbs()
        {
            var t = typeof(NumSharp.Interop.PythonNet.NDArrayPythonInterop);
            t.Assembly.GetName().Name.Should().Be("NumSharp.Interop.pythonnet");

            foreach (string verb in new[] { "ToNumpy", "ToNumpyCopy", "ToNDArray", "ToNDArrayView", "ToMemoryView" })
                t.GetMethods().Should().Contain(m => m.Name == verb, $"the verb surface includes {verb}");
        }
    }
}
