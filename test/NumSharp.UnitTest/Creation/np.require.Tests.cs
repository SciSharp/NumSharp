using System;
using AwesomeAssertions;
using NumSharp;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace NumSharp.UnitTest.Creation
{
    /// <summary>
    /// Tests for np.require — verified 1-to-1 against NumPy 2.4.2. Parses C/F/A/W/O/E requirement
    /// flags (with aliases), resolves an order ('A' default, or 'C'/'F'), then copies once (in that
    /// order) if any remaining ALIGNED/WRITEABLE/OWNDATA flag is unsatisfied. No requirements ⇒
    /// asanyarray. In NumSharp ALIGNED is always satisfied, so only WRITEABLE (broadcast views) and
    /// OWNDATA (views) can force a copy.
    /// </summary>
    [TestClass]
    public class np_require_Tests
    {
        // ─── no requirements ⇒ asanyarray ───────────────────────────────────

        [TestMethod]
        public void NoRequirements_ReturnsSameStorage()
        {
            var x = np.arange(6).reshape(2, 3);
            var r = np.require(x);
            ReferenceEquals(x.Storage, r.Storage).Should().BeTrue();
        }

        [TestMethod]
        public void NoRequirements_WithDtype_Casts()
        {
            var x = np.arange(6).reshape(2, 3);
            np.require(x, typeof(float)).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void EmptyRequirements_ReturnsSameStorage()
        {
            var x = np.arange(6).reshape(2, 3);
            ReferenceEquals(np.require(x, requirements: new string[0]).Storage, x.Storage).Should().BeTrue();
        }

        // ─── order requirements ─────────────────────────────────────────────

        [TestMethod]
        public void FortranRequirement_ProducesFContiguous()
        {
            var x = np.arange(6).astype(NPTypeCode.Int64).reshape(2, 3); // C-contig, 2-D
            var r = np.require(x, typeof(float), new[] { "A", "O", "W", "F" });
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
            r.Storage.IsView.Should().BeFalse(); // owns data
            r.Shape.IsWriteable.Should().BeTrue();
            r.dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void CAndF_Conflict_Raises() =>
            ((Action)(() => np.require(np.arange(6), requirements: new[] { "C", "F" })))
                .Should().Throw<ValueError>().WithMessage("Cannot specify both*C*F*order*");

        [TestMethod]
        public void CRequirement_OnContiguous_NoCopy()
        {
            var x = np.ascontiguousarray(np.arange(6)); // 1-D, already C-contig
            ReferenceEquals(np.require(x, requirements: new[] { "C" }).Storage, x.Storage).Should().BeTrue();
        }

        [TestMethod]
        public void FRequirement_On1D_NoCopy_Because1DisBothContiguous()
        {
            var x = np.ascontiguousarray(np.arange(6)); // 1-D is BOTH C- and F-contiguous
            ReferenceEquals(np.require(x, requirements: new[] { "F" }).Storage, x.Storage).Should().BeTrue();
        }

        // ─── OWNDATA / WRITEABLE force a copy ───────────────────────────────

        [TestMethod]
        public void OwndataRequirement_OnView_Copies()
        {
            var x = np.arange(6).reshape(2, 3); // reshape of arange is a VIEW (OWNDATA=false)
            var r = np.require(x, requirements: new[] { "O" });
            r.Storage.IsView.Should().BeFalse();       // now owns data
            ReferenceEquals(x.Storage, r.Storage).Should().BeFalse();
        }

        [TestMethod]
        public void OwndataRequirement_OnSliceView_Copies()
        {
            var v = np.arange(10)["2:8"];
            ReferenceEquals(np.require(v, requirements: new[] { "O" }).Storage, v.Storage).Should().BeFalse();
        }

        [TestMethod]
        public void WriteableRequirement_OnBroadcast_Copies()
        {
            var b = np.broadcast_to(np.arange(3), new Shape(4, 3)); // non-writeable
            var r = np.require(b, requirements: new[] { "W" });
            r.Shape.IsWriteable.Should().BeTrue();
            r.Storage.IsView.Should().BeFalse();
        }

        // ─── flag aliases & case-insensitivity ──────────────────────────────

        [TestMethod]
        public void Alias_FORTRAN_MeansF()
        {
            var x = np.arange(6).reshape(2, 3);
            np.require(x, requirements: new[] { "FORTRAN" }).Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void Alias_F_CONTIGUOUS_AsListElement_MeansF()
        {
            var x = np.arange(6).reshape(2, 3);
            np.require(x, requirements: new[] { "F_CONTIGUOUS" }).Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void EnsureArray_IsStrippedAndIgnored()
        {
            var x = np.arange(6).reshape(2, 3);
            ReferenceEquals(np.require(x, requirements: new[] { "E" }).Storage, x.Storage).Should().BeTrue();
        }

        [TestMethod]
        public void UnknownRequirement_Raises() =>
            ((Action)(() => np.require(np.arange(6), requirements: new[] { "Z" }))).Should().Throw<ValueError>();

        // ─── single-string requirements: NumPy iterates by character ────────

        [TestMethod]
        public void SingleChar_F_Works()
        {
            var x = np.arange(6).reshape(2, 3);
            np.require(x, (Type)null, "F").Shape.IsFContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void SingleString_CF_IteratesToBothOrders_Raises() =>
            ((Action)(() => np.require(np.arange(6).reshape(2, 3), (Type)null, "CF")))
                .Should().Throw<ValueError>().WithMessage("Cannot specify both*");

        [TestMethod]
        public void SingleString_MultiCharAlias_RaisesOnUnderscore() =>
            // "F_CONTIGUOUS" as a single string is iterated by char; '_' is not a flag (NumPy: KeyError).
            ((Action)(() => np.require(np.arange(6).reshape(2, 3), (Type)null, "F_CONTIGUOUS")))
                .Should().Throw<ValueError>();

        // ─── dtype convenience overloads ────────────────────────────────────

        [TestMethod]
        public void DtypeString_Overload()
        {
            var x = np.arange(6).reshape(2, 3);
            np.require(x, "float32", new[] { "C" }).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void DtypeNPTypeCode_Overload()
        {
            var x = np.arange(6).reshape(2, 3);
            np.require(x, NPTypeCode.Single).dtype.Should().Be(typeof(float));
        }

        [TestMethod]
        public void NullInput_Throws() =>
            ((Action)(() => np.require(null))).Should().Throw<ArgumentNullException>();

        // ─── second-pass edge cases (verified against NumPy) ────────────────

        [TestMethod]
        public void Like_IsAcceptedAndIgnored()
        {
            var x = np.arange(6).reshape(2, 3);
            // `like` exists for NumPy's array-function dispatch; NumSharp accepts it as a no-op.
            var r = np.require(x, typeof(float), new[] { "C" }, like: np.arange(3));
            r.dtype.Should().Be(typeof(float));
            r.Shape.IsContiguous.Should().BeTrue();
        }

        [TestMethod]
        public void FRequirement_OnCContig2D_CopiesToFContiguous()
        {
            var x = np.arange(6).astype(NPTypeCode.Int64).reshape(2, 3); // C-contig 2-D
            var r = np.require(x, requirements: new[] { "F" });
            r.Shape.IsFContiguous.Should().BeTrue();
            r.Shape.IsContiguous.Should().BeFalse();
            ReferenceEquals(r.Storage, x.Storage).Should().BeFalse(); // copied
        }

        [TestMethod]
        public void OwndataRequirement_CopiesEvenWhenOrderAlreadySatisfied()
        {
            // 'C' is satisfied (view is C-contig) but 'O' is not (it's a view) → still copies.
            var x = np.arange(6).reshape(2, 3);
            ReferenceEquals(np.require(x, requirements: new[] { "C", "O" }).Storage, x.Storage).Should().BeFalse();
        }

        [TestMethod]
        public void Aliases_LowercaseAndFullNames()
        {
            var x = np.arange(6).reshape(2, 3);
            // 'contiguous'/'c' ⇒ C (no copy on a C-contig view); 'writeable'/'aligned' are already satisfied.
            ReferenceEquals(np.require(x, requirements: new[] { "contiguous" }).Storage, x.Storage).Should().BeTrue();
            ReferenceEquals(np.require(x, requirements: new[] { "c", "writeable", "aligned" }).Storage, x.Storage).Should().BeTrue();
        }

        [TestMethod]
        public void EmptyArray_FRequirement_ProducesFContiguous()
        {
            var r = np.require(np.zeros(new Shape(0, 3), NPTypeCode.Double), requirements: new[] { "F" });
            r.Shape.IsFContiguous.Should().BeTrue();
        }
    }
}
