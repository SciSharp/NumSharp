using System;
using System.Collections.Generic;
using System.IO;
using NumSharp;

namespace NumSharp.Tests.Documentation
{
    /// <summary>
    /// Keeps the DocFX entry-point examples honest about text descriptors, CLR characters, missing values and
    /// subclass behavior; numeric storage rules must not silently become promises about packed text.
    /// </summary>
    /// <remarks>
    /// NumPy-side values were probed with NumPy 2.4.2. The C# input-kind and static-dispatch assertions also pin the
    /// deliberate language boundaries described by the guides. Every test owns a scope; disposable holders are
    /// disposed separately because a scope only reclaims NDArrays.
    /// </remarks>
    [TestClass]
    public class TextOverviewDocTests
    {
        /// <summary>Shows why a full descriptor, rather than the shared String type code, must travel with text.</summary>
        /// <exception cref="AssertFailedException">A documented descriptor, element value or metadata field differs.</exception>
        [TestMethod]
        public void Representations_KeepWidthsAndMissingValuePolicy()
        {
            using var scope = NDScope.Open();
            var unicode = np.array(new[] { "cat", "bird" });
            var fixedBytes = np.array(new[] { "cat", "bird" }, "S8");
            var variable = np.array(new[] { "cat", "bird" }, np.dtype("T"));
            Assert.AreEqual(np.dtype("U4"), unicode.dtype);
            Assert.AreEqual(16, unicode.itemsize);
            Assert.AreEqual(8, fixedBytes.itemsize);
            Assert.AreEqual(16, variable.itemsize);
            Assert.AreEqual(NPTypeCode.String, unicode.typecode);
            Assert.AreEqual(NPTypeCode.String, fixedBytes.typecode);
            Assert.AreEqual(NPTypeCode.String, variable.typecode);
            CollectionAssert.AreEqual(new byte[] { 99, 97, 116 }, (byte[])fixedBytes.item(0));

            var template = np.array(new[] { "cat", "bird" }, "U8");
            var sameType = np.ones(template.shape, template.dtype);
            var sameShapeAndType = np.ones_like(template);
            Assert.AreEqual(np.dtype("U8"), sameType.dtype);
            Assert.AreEqual(template.dtype, sameShapeAndType.dtype);
            CollectionAssert.AreEqual(new[] { "1", "1" }, NDArray.AsStringArray(sameType));

            var nullable = np.dtypes.StringDType(na_object: null);
            var notes = np.array(new object[] { "short", null }, nullable);
            notes[0] = "This text can grow without a fixed-width truncation.";
            Assert.AreEqual("This text can grow without a fixed-width truncation.", notes.item(0));
            Assert.IsNull(notes.item(1));
            Assert.AreEqual(nullable, np.empty_like(notes).dtype);
            // Cell accounting deliberately excludes the separately allocated UTF-8 payload.
            Assert.AreEqual(32L, notes.nbytes);
        }

        /// <summary>Prevents the guides from treating a C# char or a direct byte array as a NumPy string scalar.</summary>
        /// <exception cref="AssertFailedException">A documented C# conversion or descriptor property differs.</exception>
        [TestMethod]
        public void CSharpInputs_DistinguishTextCodeUnitsAndByteArrays()
        {
            using var scope = NDScope.Open();
            NDArray textScalar = "cat";
            Assert.AreEqual(0, textScalar.ndim);
            Assert.AreEqual(np.dtype("U3"), textScalar.dtype);
            Assert.AreEqual("cat", textScalar.item());

            var codeUnits = np.array(new[] { 'a', 'b' });
            Assert.AreEqual(NPTypeCode.Char, codeUnits.typecode);
            Assert.AreEqual(2, codeUnits.dtype.itemsize);
            Assert.AreEqual('u', codeUnits.dtype.kind);
            CollectionAssert.AreEqual(new[] { "97", "98" }, NDArray.AsStringArray(codeUnits.astype("U")));
            Assert.AreEqual(16, np.dtype(typeof(decimal)).itemsize);
            Assert.AreEqual('b', np.bool_.kind);
            Assert.AreEqual('?', np.bool_.@char);
            Assert.AreEqual(2, ((DType)np.@char).itemsize);

            byte[] bytes = { 97, 98 };
            Assert.AreEqual(NPTypeCode.Byte, np.array(bytes).typecode);
            Assert.AreEqual(np.dtype("S2"), NDArray.Scalar((object)bytes).dtype);
            Assert.AreEqual(np.dtype("S2"), np.array(new object[] { bytes }).dtype);
            CollectionAssert.AreEqual(new[] { 1.0, 2.0, 3.0 }, np.fromstring("1, 2, 3", sep: ",").ToArray<double>());
        }

        /// <summary>Executes the text quickstarts, including weighted selection without pretending interpolation is valid.</summary>
        /// <exception cref="AssertFailedException">A documented value, dtype or expected refusal differs.</exception>
        [TestMethod]
        public void Quickstarts_SortCountRepeatAndSelectText()
        {
            using var scope = NDScope.Open();
            var labels = np.array(new[] { "pear", "fig", "apple", "fig" });
            var sorted = np.sort(labels);
            var (values, counts) = np.unique(labels, return_counts: true);
            var repeated = np.strings.multiply(labels, 2);
            CollectionAssert.AreEqual(new[] { "apple", "fig", "fig", "pear" }, NDArray.AsStringArray(sorted));
            CollectionAssert.AreEqual(new[] { "apple", "fig", "pear" }, NDArray.AsStringArray(values));
            CollectionAssert.AreEqual(new long[] { 1, 2, 1 }, counts.ToArray<long>());
            CollectionAssert.AreEqual(new[] { "pearpear", "figfig", "appleapple", "figfig" }, NDArray.AsStringArray(repeated));
            Assert.AreEqual("pear", labels.item(0));

            var names = np.array(new[] { "cat", "dog", "bird" });
            CollectionAssert.AreEqual(new[] { true, false, false }, (names == "cat").ToArray<bool>());
            CollectionAssert.AreEqual(new[] { "bird", "cat", "dog" }, NDArray.AsStringArray(np.sort(names)));
            CollectionAssert.AreEqual(new[] { "catcat", "dogdog", "birdbird" }, NDArray.AsStringArray(np.strings.multiply(names, 2)));
            var growing = names.astype(np.dtypes.StringDType());
            growing[0] = "a much longer label";
            Assert.AreEqual("a much longer label", growing.item(0));
            Assert.AreEqual("cat", names.item(0));

            var pair = np.array(new[] { "cat", "dog" });
            CollectionAssert.AreEqual(new[] { "cat!", "dog!" }, NDArray.AsStringArray(np.add(pair, "!")));
            Assert.AreEqual("catdog", np.sum(pair.astype(np.dtypes.StringDType())).item());
            Assert.ThrowsException<TypeError>(() => np.sum(pair));

            var fruit = np.array(new[] { "pear", "fig", "apple" });
            var weights = np.array(new[] { 1.0, 3.0, 1.0 });
            Assert.AreEqual("fig", np.quantile(fruit, 0.5, method: "inverted_cdf", weights: weights).item());
            Assert.AreEqual("fig", np.quantile(fruit.astype("T"), 0.5, method: "inverted_cdf", weights: weights).item());
        }

        /// <summary>Distinguishes Boolean conversion and binary logical loops from NumPy's missing unary text loop.</summary>
        /// <exception cref="AssertFailedException">A documented truth value or logical-not refusal differs.</exception>
        [TestMethod]
        public void TextTruth_DoesNotImplyALogicalNotLoop()
        {
            using var scope = NDScope.Open();
            foreach (DType dtype in new DType[] { "U2", "S2", "T" })
            {
                var text = np.array(new[] { "a", "" }, dtype);
                CollectionAssert.AreEqual(new[] { true, false }, text.astype(np.bool_).ToArray<bool>());
                CollectionAssert.AreEqual(new[] { true, false }, np.logical_and(text, text).ToArray<bool>());
                CollectionAssert.AreEqual(new[] { true, false }, np.logical_or(text, text).ToArray<bool>());
                CollectionAssert.AreEqual(new[] { false, false }, np.logical_xor(text, text).ToArray<bool>());
                Assert.ThrowsException<TypeError>(() => np.logical_not(text));
            }
        }

        /// <summary>Pins the distinction between an NA that propagates through concatenation and one ordered above text.</summary>
        /// <exception cref="AssertFailedException">The documented NA comparisons or reduction selections differ.</exception>
        [TestMethod]
        public void MissingValues_MinimumDoesNotPropagateANanLikeNa()
        {
            using var scope = NDScope.Open();
            var nan = np.dtypes.StringDType(na_object: double.NaN);
            var text = np.array(new object[] { "b", double.NaN, "a", double.NaN }, nan);
            Assert.AreEqual("a", np.min(text).item());
            Assert.IsTrue(double.IsNaN((double)np.max(text).item()));
            Assert.IsTrue(double.IsNaN((double)np.sum(text).item()));
            Assert.AreEqual(3L, np.argmax(text));
            CollectionAssert.AreEqual(new[] { false, true, false, true }, np.isnan(text).ToArray<bool>());
            CollectionAssert.AreEqual(new[] { false, false, false, false }, (text != text).ToArray<bool>());
        }

        /// <summary>Checks that ordinary text access, chararray stripping and static dispatch remain distinct.</summary>
        /// <exception cref="AssertFailedException">A documented store, stripped scalar or result class differs.</exception>
        [TestMethod]
        public void CharArrays_AreNotCharDtypesOrPlainStringArrays()
        {
            using var scope = NDScope.Open();
            var names = np.array(new[] { "cat", "dog" }, "U4");
            Assert.AreEqual("cat", (string)names.item(0));
            names[1] = "zebras";
            Assert.AreEqual("zebr", names.item(1));
            var text = names.astype(np.dtypes.StringDType(na_object: null));
            text[1] = null;
            Assert.IsNull(text.item(1));

            var c = np.@char.array(new[] { "cat  ", "dog" });
            Assert.AreEqual("cat", c[0].item());
            Assert.AreEqual("cat  ", c.item(0));
            CollectionAssert.AreEqual(new[] { true, false }, (c == "cat").ToArray<bool>());
            NDArray plain = c;
            CollectionAssert.AreEqual(new[] { false, false }, (plain == "cat").ToArray<bool>());
            Assert.IsInstanceOfType(np.reshape(plain, new Shape(1, 2)), typeof(NDCharArray));
            Assert.IsFalse(np.concatenate(new NDArray[] { c, c }) is NDCharArray);
            // Split results own more than their NDArrays, so they need explicit disposal even inside this scope.
            using NDListArray pieces = np.@char.split(np.array(new[] { "a b", "c" }));
            CollectionAssert.AreEqual(new List<object> { "a", "b" }, (List<object>)pieces.item(0));
        }

        /// <summary>Exercises a reference-aware iterator and the supported persistence route instead of copying raw cells.</summary>
        /// <exception cref="AssertFailedException">A missing value or StringDType descriptor is lost across iteration or I/O.</exception>
        [TestMethod]
        public void StringDType_IteratesAndPersistsValuesNotRawCells()
        {
            using var scope = NDScope.Open();
            var text = np.array(new object[] { "cat", null }, np.dtypes.StringDType(na_object: null));
            var seen = new List<object>();
            using (var it = np.nditer(text, flags: new[] { "refs_ok" }))
                foreach (var operands in it)
                    seen.Add(operands[0].item());
            CollectionAssert.AreEqual(new List<object> { "cat", null }, seen);

            using var stream = new MemoryStream();
            np.save(stream, text);
            stream.Position = 0;
            var copy = np.load_npy(stream, allow_pickle: true);
            Assert.AreEqual(text.dtype, copy.dtype);
            Assert.AreEqual("cat", copy.item(0));
            Assert.IsNull(copy.item(1));
            Assert.IsFalse(np.shares_memory(copy, text));
        }

        /// <summary>Checks descriptor aliases and promotion so examples cannot lose the width by going through System.Type.</summary>
        /// <exception cref="AssertFailedException">An alias, promotion or required rejection differs from the guide.</exception>
        [TestMethod]
        public void Descriptors_AreThePublicDtypeCurrency()
        {
            DType integers = np.int32;
            DType unicode = np.str_;
            DType bytes = np.bytes_;
            DType text = np.dtypes.StringDType(na_object: null);
            Assert.AreEqual(typeof(int), integers.type);
            Assert.AreEqual(np.dtype("U"), unicode);
            Assert.AreEqual(np.dtype("S"), bytes);
            Assert.AreEqual('T', text.kind);
            Assert.AreNotEqual(np.dtype("U3"), np.dtype("U5"));
            Assert.AreEqual(np.dtype("U3"), np.promote_types(np.dtype("S3"), np.dtype("U2")));
            Assert.AreEqual(np.dtype("T"), np.promote_types(np.dtype("T"), np.dtype("U2")));
            Assert.ThrowsException<DTypePromotionError>(() => np.promote_types(np.dtype("T"), np.int32));
        }

        /// <summary>Demonstrates heterogeneous tables as separate text and numeric columns rather than unsupported records.</summary>
        /// <exception cref="AssertFailedException">The documented column value or text-file projection differs.</exception>
        [TestMethod]
        public void TextColumns_CoexistWithNumericColumns()
        {
            using var scope = NDScope.Open();
            var names = np.array(new[] { "Rex", "Fido" });
            var age = np.array(new[] { 9, 3 });
            var weight = np.array(new[] { 81.0f, 27.0f });
            Assert.AreEqual("Rex", names.item(0));
            Assert.AreEqual(9, age.item(0));
            Assert.AreEqual(81.0f, weight.item(0));

            var lines = new[] { "Rex,9", "Fido,3" };
            var labels = np.loadtxt(lines, dtype: np.dtypes.StringDType(), delimiter: ",", usecols: new[] { 0 });
            var ages = np.loadtxt(lines, dtype: np.int32, delimiter: ",", usecols: new[] { 1 });
            CollectionAssert.AreEqual(new[] { "Rex", "Fido" }, NDArray.AsStringArray(labels));
            CollectionAssert.AreEqual(new[] { 9, 3 }, ages.ToArray<int>());
        }
    }
}
