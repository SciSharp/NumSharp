namespace NumSharp.Tests.Lifetime
{
    /// <summary>
    ///     Pins <see cref="NDArray.ReplaceData(NDArray)"/> and its overloads to the ARC contract every array
    ///     keeps: it holds exactly ONE counted reference on <c>Storage.InternalArray</c>, so a storage-level
    ///     buffer swap must MOVE that reference — take one on the adopted buffer, release the one on the old.
    /// </summary>
    /// <remarks>
    ///     Before the fix the swap moved nothing: the old buffer stayed referenced forever (stranded until a
    ///     GC finalized it), and <see cref="NDArray.Dispose()"/> released a reference on the ADOPTED buffer that the
    ///     array never took — for <c>x.ReplaceData(nd)</c> that was one release of <c>nd</c>'s buffer too many,
    ///     freeing it under a live <c>nd</c>. Each test asserts both directions through the refcount itself
    ///     (<c>IsUniquelyReferenced</c> / <c>IsReleased</c>), which observes the defect deterministically where a
    ///     read-back of freed-but-not-yet-reused memory would not.
    /// </remarks>
    [TestClass]
    public class ReplaceDataArcTests
    {
        /// <summary>
        ///     Adopting another array's buffer shares it: both arrays hold a reference, the old buffer (this
        ///     array was its only owner) is released at once, and disposing the adopter leaves the donor the sole,
        ///     still-live owner of intact data.
        /// </summary>
        [TestMethod]
        public void ReplaceData_NDArray_MovesTheReferenceOntoTheSharedBuffer()
        {
            using var donor = np.arange(6.0);
            var x = np.zeros(new Shape(6));
            var old = x.Storage.InternalArray;

            x.ReplaceData(donor);

            old.IsReleased.Should().BeTrue("the adopter was the old buffer's only owner, so its reference moved off it");
            donor.Storage.InternalArray.IsUniquelyReferenced.Should().BeFalse("the adopter now holds its OWN reference on the donor's buffer");
            x.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);

            x.Dispose();

            donor.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue("disposing the adopter released only the reference it took");
            LeakGuards.StillUsable(donor);
            donor.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0, 5.0);
        }

        /// <summary>
        ///     Replacing the buffer with a MANAGED array (wrapped, never pooled) still releases the array's reference
        ///     on its old pooled buffer, and the array reads the new values.
        /// </summary>
        [TestMethod]
        public void ReplaceData_ManagedArray_ReleasesTheOldBuffer()
        {
            using var x = np.zeros(new Shape(3));
            var old = x.Storage.InternalArray;

            x.ReplaceData(new double[] { 7, 8, 9 });

            old.IsReleased.Should().BeTrue("the array's one reference moved off its old buffer");
            x.ToArray<double>().Should().Equal(7.0, 8.0, 9.0);
        }

        /// <summary>
        ///     A bare unmanaged slice (e.g. from <see cref="NDArray.CloneData()"/>) starts with no owner; adopting it
        ///     makes the array its owner, so it lives exactly as long as the array — freed with it, not before.
        /// </summary>
        [TestMethod]
        public void ReplaceData_BareSlice_IsOwnedByTheAdopterAndFreedWithIt()
        {
            using var source = np.arange(4.0);
            var slice = source.CloneData();
            var x = np.zeros(new Shape(4));
            var old = x.Storage.InternalArray;

            x.ReplaceData(slice);

            old.IsReleased.Should().BeTrue();
            slice.IsReleased.Should().BeFalse("the adopter now owns the slice");
            x.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0);

            x.Dispose();

            slice.IsReleased.Should().BeTrue("the slice's only owner was disposed");
            LeakGuards.StillUsable(source);
        }

        /// <summary>
        ///     Swapping onto the buffer the array already references nets out: the reference is added before the
        ///     old one is released, so the count never touches zero and the array stays fully usable.
        /// </summary>
        [TestMethod]
        public void ReplaceData_SelfSwap_NeverFreesTheBufferUnderTheArray()
        {
            using var x = np.arange(5.0);

            x.ReplaceData(x);

            LeakGuards.StillUsable(x);
            x.Storage.InternalArray.IsUniquelyReferenced.Should().BeTrue();
            x.ToArray<double>().Should().Equal(0.0, 1.0, 2.0, 3.0, 4.0);
        }
    }
}
