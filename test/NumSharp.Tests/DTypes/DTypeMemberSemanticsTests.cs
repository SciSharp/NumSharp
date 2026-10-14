using System;
using System.Collections.Generic;

namespace NumSharp.Tests.DTypes
{
    /// <summary>
    ///     Readable pins for the <c>numpy.dtype</c> / DType-class behaviours the NumPy-name pass ported or corrected: the
    ///     explicit host byte order <c>newbyteorder</c> keeps, the parser's fold of it, <c>len</c> / <c>dtype[key]</c> /
    ///     <c>isalignedstruct</c>, the <c>align</c> / <c>copy</c> / <c>metadata</c> keywords of <c>np.dtype</c>, the
    ///     identity rules of promotion and canonicalisation, and the NumPy C names of the byte-order characters. Every
    ///     expected value was probed against NumPy 2.4.2; <see cref="DTypeOracleTests"/> replays the full matrix.
    /// </summary>
    [TestClass]
    public class DTypeMemberSemanticsTests
    {
        /// <summary>The host's explicit byte-order character (<c>'&lt;'</c> on every host CI runs on).</summary>
        private static readonly char Host = BitConverter.IsLittleEndian ? '<' : '>';

        /// <summary>The opposite byte-order character.</summary>
        private static readonly char Opposite = BitConverter.IsLittleEndian ? '>' : '<';

        /// <summary>
        ///     <c>newbyteorder('&lt;')</c> on a little-endian host keeps an EXPLICIT host order (NumPy's
        ///     <c>PyArray_DescrNewByteorder</c> stores the requested character): native, equal to and hashing like the
        ///     <c>'='</c> singleton, printed by <c>str</c> as the name but by <c>repr</c> in the short form, and a copy.
        /// </summary>
        [TestMethod]
        public void NewByteorder_KeepsAnExplicitHostOrder()
        {
            var explicitHost = np.dtype("f8").newbyteorder(Host.ToString());
            explicitHost.byteorder.Should().Be(Host);
            explicitHost.isnative.Should().BeTrue();
            explicitHost.Should().Be(DType.Double);
            explicitHost.GetHashCode().Should().Be(DType.Double.GetHashCode());
            explicitHost.ToString().Should().Be("float64");
            explicitHost.ToString(true).Should().Be($"dtype('{Host}f8')");
            explicitHost.str.Should().Be($"{Host}f8");
            explicitHost.isbuiltin.Should().Be(0);
        }

        /// <summary>
        ///     Swapping a non-native descriptor gives the host's EXPLICIT order, not <c>'='</c> (former surface gap SU17):
        ///     <c>repr(np.dtype('&gt;i4').newbyteorder())</c> is <c>dtype('&lt;i4')</c>; swapping a native one gives the opposite order.
        /// </summary>
        [TestMethod]
        public void NewByteorder_SwapOfANonNativeDescriptor_IsTheExplicitHostOrder()
        {
            np.dtype($"{Opposite}i4").newbyteorder().ToString(true).Should().Be($"dtype('{Host}i4')");
            np.dtype($"{Opposite}c16").newbyteorder().byteorder.Should().Be(Host);
            np.dtype("i4").newbyteorder().byteorder.Should().Be(Opposite);
            np.dtype("i4").newbyteorder("S").newbyteorder("S").byteorder.Should().Be(Host, "the second swap leaves the explicit host order");
            np.dtype($"{Opposite}f8").newbyteorder("|").byteorder.Should().Be(Opposite, "'|' leaves the byte order as it is");
            np.dtype("?").newbyteorder(">").byteorder.Should().Be('|', "a single-byte class keeps '|'");
            np.dtype("?").newbyteorder(">").isbuiltin.Should().Be(0, "newbyteorder always returns a copy");
        }

        /// <summary>
        ///     NumPy's byte-order request errors: an unknown first character is ValueError quoting the WHOLE string with
        ///     Python's repr (<c>'Xyz'</c>, <c>"'"</c>); an empty one too; null (NumPy's <c>None</c>) is TypeError.
        /// </summary>
        [TestMethod]
        public void NewByteorder_Errors_AreNumPysTexts()
        {
            var d = np.dtype("f8");
            ((Action)(() => d.newbyteorder("Xyz"))).Should().Throw<ValueError>().WithMessage("byteorder not recognized (got 'Xyz')");
            ((Action)(() => d.newbyteorder(""))).Should().Throw<ValueError>().WithMessage("byteorder not recognized (got '')");
            ((Action)(() => d.newbyteorder("'"))).Should().Throw<ValueError>().WithMessage("byteorder not recognized (got \"'\")");
            ((Action)(() => d.newbyteorder((string)null))).Should().Throw<TypeError>().WithMessage("byteorder must be str, not NoneType");
            d.newbyteorder("lx").byteorder.Should().Be('<', "only the first character is read");
        }

        /// <summary>
        ///     The parser folds an explicit HOST order into <c>'='</c> (NumPy's <c>_convert_from_str</c>), so
        ///     <c>np.dtype('&lt;f8')</c> is the float64 singleton and <c>np.dtype('&lt;M8[ns]')</c> stores <c>'='</c>.
        /// </summary>
        [TestMethod]
        public void Parser_FoldsAnExplicitHostOrder()
        {
            np.dtype($"{Host}f8").Should().BeSameAs(DType.Double);
            np.dtype($"{Host}M8[ns]").byteorder.Should().Be('=');
            np.dtype($"{Opposite}M8[ns]").byteorder.Should().Be(Opposite);
            np.dtype("M").Should().BeSameAs(np.dtype("M"), "the datetime letter is the builtin singleton");
            np.dtype("M").isbuiltin.Should().Be(1);
            np.dtype("M8").isbuiltin.Should().Be(0, "the datetime typestr builds a fresh descriptor");
        }

        /// <summary>
        ///     The field protocol of a non-structured dtype: <c>len(dtype)</c> is 0, <c>dtype[key]</c> raises NumPy's
        ///     KeyError for any key (the message is Python's <c>str(KeyError)</c>, quoted, naming <c>str(dtype)</c>), and
        ///     <c>isalignedstruct</c> is False.
        /// </summary>
        [TestMethod]
        public void FieldProtocol_OfANonStructuredDtype()
        {
            np.dtype("f8").__len__().Should().Be(0);
            np.dtype("f8").isalignedstruct.Should().BeFalse();
            ((Action)(() => np.dtype("f8").__getitem__(0))).Should().Throw<KeyError>()
                .WithMessage("'There are no fields in dtype float64.'");
            ((Action)(() => np.dtype(">f8").__getitem__("x"))).Should().Throw<KeyError>()
                .WithMessage("'There are no fields in dtype >f8.'");
            ((Action)(() => np.dtype("M8[ns]").__getitem__(null))).Should().Throw<KeyError>()
                .WithMessage("'There are no fields in dtype datetime64[ns].'");
        }

        /// <summary>
        ///     <c>np.dtype(obj, align, copy, metadata)</c>: <c>align</c> changes nothing for these dtypes; <c>copy</c>
        ///     copies only a builtin singleton (an already-fresh descriptor comes back as itself); <c>metadata</c> always
        ///     gives a fresh descriptor whose metadata merges into the existing dict without overwriting its keys, and is
        ///     ignored by equality, hashing and printing.
        /// </summary>
        [TestMethod]
        public void DtypeKeywords_AlignCopyMetadata()
        {
            np.dtype("f8", align: true).Should().BeSameAs(DType.Double);

            var copy = np.dtype("f8", copy: true);
            copy.Should().NotBeSameAs(DType.Double);
            copy.Should().Be(DType.Double);
            copy.isbuiltin.Should().Be(0);
            var fresh = np.dtype(">f8");
            np.dtype(fresh, copy: true).Should().BeSameAs(fresh, "only the static builtin descriptor is copied");

            var withMeta = np.dtype("f8", metadata: new Dictionary<string, object> { ["a"] = 1L });
            withMeta.metadata.Should().ContainKey("a").WhoseValue.Should().Be(1L);
            withMeta.Should().Be(DType.Double);
            withMeta.GetHashCode().Should().Be(DType.Double.GetHashCode());
            withMeta.ToString(true).Should().Be("dtype('float64')");
            withMeta.isbuiltin.Should().Be(0);
            DType.Double.metadata.Should().BeNull();
            np.dtype("f8", metadata: new Dictionary<string, object>()).metadata.Should().BeEmpty("an empty attachment is an empty dict, not None");

            var merged = np.dtype(withMeta, metadata: new Dictionary<string, object> { ["a"] = 5L, ["c"] = 3L });
            merged.metadata["a"].Should().Be(1L, "PyDict_Merge(override=0) keeps the existing key");
            merged.metadata["c"].Should().Be(3L);
            withMeta.newbyteorder("S").metadata.Should().ContainKey("a", "newbyteorder copies keep the metadata");
        }

        /// <summary>
        ///     <c>np.promote_types</c>'s fast path is POINTER identity: the same native descriptor twice comes back as
        ///     itself (metadata kept), two equal but distinct descriptors go through the common class and come back as the
        ///     singleton (metadata and an explicit host order dropped).
        /// </summary>
        [TestMethod]
        public void PromoteTypes_FastPathIsIdentity()
        {
            var withMeta = np.dtype("f8", metadata: new Dictionary<string, object> { ["a"] = 1L });
            np.promote_types(withMeta, withMeta).Should().BeSameAs(withMeta);
            np.promote_types(withMeta, np.dtype("f8", metadata: new Dictionary<string, object> { ["a"] = 1L }))
                .Should().BeSameAs(DType.Double);
            np.promote_types(np.dtype("f8").newbyteorder(Host.ToString()), DType.Double).Should().BeSameAs(DType.Double);
        }

        /// <summary>
        ///     The class slot <c>ensure_canonical</c> returns a native descriptor itself and a non-native one as a fresh
        ///     <c>'='</c> copy — the copy keeps the metadata; the singleton is never handed out for it.
        /// </summary>
        [TestMethod]
        public void EnsureCanonical_CopiesANonNativeDescriptor()
        {
            var big = np.dtype($"{Opposite}f8", metadata: new Dictionary<string, object> { ["k"] = "v" });
            var canon = big.Meta.ensure_canonical(big);
            canon.byteorder.Should().Be('=');
            canon.Should().NotBeSameAs(DType.Double);
            canon.metadata["k"].Should().Be("v");
            np.result_type(new[] { np.dtype($"{Opposite}f8") }).isbuiltin.Should().Be(0, "NumPy 2.4.2: np.result_type(np.dtype('>f8')).isbuiltin == 0");
            Action foreign = () => np.dtypes.Float64DType.ensure_canonical(DType.Int32);
            foreign.Should().Throw<ArgumentException>();
        }

        /// <summary>The byte-order characters under NumPy's C macro names (<c>ndarraytypes.h</c>).</summary>
        [TestMethod]
        public void ByteOrderMacros_AreNumPysCharacters()
        {
            DType.NPY_LITTLE.Should().Be('<');
            DType.NPY_BIG.Should().Be('>');
            DType.NPY_NATIVE.Should().Be('=');
            DType.NPY_SWAP.Should().Be('s');
            DType.NPY_IGNORE.Should().Be('|');
            DType.NPY_NATBYTE.Should().Be(Host);
            DType.NPY_OPPBYTE.Should().Be(Opposite);
        }
    }
}
