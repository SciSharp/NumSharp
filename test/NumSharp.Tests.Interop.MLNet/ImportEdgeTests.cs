using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.MLNet;

namespace NumSharp.Tests.Interop.MLNet
{
    /// <summary>Materializing columns out of REAL ML.NET data views (not just NumSharp-backed ones).</summary>
    [TestClass]
    public class ImportEdgeTests : MLNetTestBase
    {
        // ---- reading a non-NumSharp IDataView (POCO loader) -----------------------------------------

        [TestMethod]
        public void ScalarColumns_FromPocoLoader_CrossCheck()
        {
            var rows = new[]
            {
                new ScalarRow { A = 1f, B = 10f, Flag = true },
                new ScalarRow { A = 2f, B = 20f, Flag = false },
                new ScalarRow { A = 3f, B = 30f, Flag = true },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);

            using NDArray a = dv.ToNDArray("A");
            using NDArray b = dv.ToNDArray("B");
            using NDArray flag = dv.ToNDArray("Flag");
            AssertShape(a, 3);
            Assert.AreEqual(NPTypeCode.Single, a.typecode);
            Assert.AreEqual(NPTypeCode.Boolean, flag.typecode);
            CollectionAssert.AreEqual(new float[] { 1, 2, 3 }, a.ToArray<float>());
            CollectionAssert.AreEqual(new float[] { 10, 20, 30 }, b.ToArray<float>());
            CollectionAssert.AreEqual(new[] { true, false, true }, flag.ToArray<bool>());
        }

        [TestMethod]
        public void VectorColumn_FixedSize_FromPocoLoader_Is2D()
        {
            var rows = new[]
            {
                new VecRow { F = new float[] { 1, 2, 3 } },
                new VecRow { F = new float[] { 4, 5, 6 } },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 2, 3);
            Assert.AreEqual(6f, (float)back[1, 2]);
        }

        [TestMethod]
        public void VectorColumn_UnsizedButConsistent_Materializes()
        {
            var rows = new[]
            {
                new UnsizedVecRow { F = new float[] { 1, 2 } },
                new UnsizedVecRow { F = new float[] { 3, 4 } },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            using NDArray back = dv.ToNDArray("F");
            AssertShape(back, 2, 2);
        }

        /// <summary>
        ///     A vector column whose rows disagree on length is refused (NumPy refuses ragged rows too), and the message's
        ///     route - ML.NET's <c>GetColumn</c> of the item type's array, made one length before <c>np.array</c> - reads
        ///     it as written. The message used to suggest "an ML.NET vector-size transform", which ML.NET does not have.
        /// </summary>
        [TestMethod]
        public void VectorColumn_ActuallyVariableLength_Throws()
        {
            var rows = new[]
            {
                new UnsizedVecRow { F = new float[] { 1, 2 } },
                new UnsizedVecRow { F = new float[] { 1, 2, 3 } },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            ArgumentException ex = Assert.ThrowsException<ArgumentException>(() => dv.ToNDArray("F"));
            StringAssert.Contains(ex.Message, "variable-length");
            StringAssert.Contains(ex.Message, "view.GetColumn<Single[]>(\"F\")");
            Assert.IsFalse(ex.Message.Contains("vector-size transform"), ex.Message);

            // The route: the ragged rows, padded with zeros to the longest, into one rectangular array.
            float[][] ragged = dv.GetColumn<Single[]>("F").ToArray();
            int width = ragged.Max(r => r.Length);
            using NDArray padded = np.array(ragged.Select(r => r.Concat(new float[width - r.Length])).SelectMany(v => v).ToArray()).reshape(ragged.Length, width);
            AssertShape(padded, 2, 3);
            CollectionAssert.AreEqual(new float[] { 1, 2, 0, 1, 2, 3 }, padded.ToArray<float>());
        }

        /// <summary>
        ///     A text column is refused, and the message's route - ML.NET's <c>GetColumn&lt;string&gt;</c> with this
        ///     column's name, into <c>np.array</c> - reads it as a fixed-width <c>U</c> array, as written.
        /// </summary>
        [TestMethod]
        public void TextColumn_Throws()
        {
            var rows = new[] { new TextRow { T = "hello" }, new TextRow { T = "wörld!" } };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(() => dv.ToNDArray("T"));
            StringAssert.Contains(ex.Message, "text");
            StringAssert.Contains(ex.Message, "np.array(view.GetColumn<string>(\"T\").ToArray())");

            using NDArray read = np.array(dv.GetColumn<string>("T").ToArray());
            Assert.AreEqual("<U6", read.dtype.str);
            CollectionAssert.AreEqual(new[] { "hello", "wörld!" }, NDArray.AsStringArray(read));
        }

        /// <summary>
        ///     A VECTOR-of-text column (<c>TokenizeIntoWords</c>) is refused with its own route: ML.NET reads such a
        ///     column as <c>VBuffer&lt;ReadOnlyMemory&lt;char&gt;&gt;</c> and refuses the scalar getter the message used
        ///     to name ("Invalid TValue"), so the message names <c>GetColumn&lt;string[]&gt;</c>, which reads it as a
        ///     <c>(rows, width)</c> <c>U</c> array when the rows have one length.
        /// </summary>
        [TestMethod]
        public void VectorOfTextColumn_Throws_AndNamesTheVectorRoute()
        {
            var rows = new[] { new TextRow { T = "the quick fox" }, new TextRow { T = "a lazy dog" } };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            IDataView tokens = Ml.Transforms.Text.TokenizeIntoWords("Tokens", "T").Fit(dv).Transform(dv);
            NotSupportedException ex = Assert.ThrowsException<NotSupportedException>(() => tokens.ToNDArray("Tokens"));
            StringAssert.Contains(ex.Message, "np.array(view.GetColumn<string[]>(\"Tokens\").ToArray())");

            DataViewSchema.Column column = tokens.Schema["Tokens"];
            using (DataViewRowCursor cursor = tokens.GetRowCursor(new[] { column }))
                Assert.ThrowsException<InvalidOperationException>(() => cursor.GetGetter<ReadOnlyMemory<char>>(column), "ML.NET refuses the scalar getter on a vector column");

            using NDArray read = np.array(tokens.GetColumn<string[]>("Tokens").ToArray());
            CollectionAssert.AreEqual(new long[] { 2, 3 }, read.shape);
            CollectionAssert.AreEqual(new[] { "the", "quick", "fox", "a", "lazy", "dog" }, NDArray.AsStringArray(read.flatten()));
        }

        /// <summary>
        ///     A scalar TIME column (a POCO's or a DataFrame's <c>DateTime</c>, ML.NET's <c>DateTimeOffset</c> /
        ///     <c>TimeSpan</c>) is refused with its route - ML.NET's <c>GetColumn</c> of the column's raw type, keeping the
        ///     ticks (<c>UtcTicks</c> for <c>DateTimeOffset</c>, so rows of different offsets share one clock) - and the route
        ///     reads it as an int64 array, as written. The message also says which columns the bridge maps; it used to claim
        ///     "NumSharp carries bool, the eight integer types, float and double", which is false of NumSharp.
        /// </summary>
        [TestMethod]
        public void TimeColumns_Throw_AndNameTheTicksRoute()
        {
            var rows = new[]
            {
                new TimeRow { When = new DateTime(2026, 11, 3, 4, 30, 0), At = new DateTimeOffset(2026, 11, 3, 4, 30, 0, TimeSpan.FromHours(2)), Took = TimeSpan.FromMilliseconds(1.5) },
                new TimeRow { When = DateTime.MinValue, At = new DateTimeOffset(2026, 11, 3, 2, 30, 0, TimeSpan.Zero), Took = TimeSpan.FromDays(-1) },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);

            NotSupportedException when = Assert.ThrowsException<NotSupportedException>(() => dv.ToNDArray("When"));
            StringAssert.Contains(when.Message, "np.array(view.GetColumn<DateTime>(\"When\").Select(v => v.Ticks).ToArray())");
            StringAssert.Contains(when.Message, "this bridge maps ML.NET's boolean, integer");
            Assert.IsFalse(when.Message.Contains("NumSharp carries"), when.Message);
            using (NDArray read = np.array(dv.GetColumn<DateTime>("When").Select(v => v.Ticks).ToArray()))
            {
                Assert.AreEqual(NPTypeCode.Int64, read.typecode);
                CollectionAssert.AreEqual(rows.Select(r => r.When.Ticks).ToArray(), read.ToArray<long>());
            }

            NotSupportedException at = Assert.ThrowsException<NotSupportedException>(() => dv.ToNDArray("At"));
            StringAssert.Contains(at.Message, "np.array(view.GetColumn<DateTimeOffset>(\"At\").Select(v => v.UtcTicks).ToArray())");
            using (NDArray read = np.array(dv.GetColumn<DateTimeOffset>("At").Select(v => v.UtcTicks).ToArray()))
                Assert.AreEqual(read.GetInt64(0), read.GetInt64(1), "the two rows are one instant at two offsets");

            NotSupportedException took = Assert.ThrowsException<NotSupportedException>(() => dv.ToNDArray("Took"));
            StringAssert.Contains(took.Message, "np.array(view.GetColumn<TimeSpan>(\"Took\").Select(v => v.Ticks).ToArray())");
            using (NDArray read = np.array(dv.GetColumn<TimeSpan>("Took").Select(v => v.Ticks).ToArray()))
                CollectionAssert.AreEqual(new[] { 15_000L, TimeSpan.FromDays(-1).Ticks }, read.ToArray<long>());
        }

        // ---- key column -----------------------------------------------------------------------------

        [TestMethod]
        public void KeyColumn_ReadsAsUInt32()
        {
            var rows = new[] { new TextRow { T = "a" }, new TextRow { T = "b" }, new TextRow { T = "a" } };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            ITransformer model = Ml.Transforms.Conversion.MapValueToKey("K", "T").Fit(dv);
            IDataView keyed = model.Transform(dv);

            Assert.IsInstanceOfType(keyed.Schema["K"].Type, typeof(KeyDataViewType));
            using NDArray k = keyed.ToNDArray("K");
            Assert.AreEqual(NPTypeCode.UInt32, k.typecode);
            // keys are 1-based; "a"->1, "b"->2, "a"->1
            CollectionAssert.AreEqual(new uint[] { 1, 2, 1 }, k.ToArray<uint>());
        }

        // ---- empty / lazy views ---------------------------------------------------------------------

        [TestMethod]
        public void EmptyView_ScalarColumn_IsZeroRows()
        {
            using NDArray a = np.arange(10).astype(NPTypeCode.Single).reshape(10, 1);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            IDataView empty = Ml.Data.FilterRowsByColumn(dv, "v", lowerBound: 100, upperBound: 200);   // removes all
            using NDArray back = empty.ToNDArray("v");
            AssertShape(back, 0);
        }

        [TestMethod]
        public void LazyView_UnknownRowCount_Materializes()
        {
            using NDArray a = np.arange(10).astype(NPTypeCode.Single).reshape(10, 1);
            using NDArrayDataView dv = a.AsDataView(new[] { "v" });
            IDataView filtered = Ml.Data.FilterRowsByColumn(dv, "v", lowerBound: 3, upperBound: 8);   // keeps 3..7
            Assert.IsNull(filtered.GetRowCount(), "a filtered view reports an unknown row count");
            using NDArray back = filtered.ToNDArray("v");
            CollectionAssert.AreEqual(new float[] { 3, 4, 5, 6, 7 }, back.ToArray<float>());
        }

        // ---- overloads & multiple columns -----------------------------------------------------------

        [TestMethod]
        public void ToNDArray_ByColumnStruct_Overload()
        {
            using NDArray a = Arange(NPTypeCode.Single, 3, 4);
            using NDArrayDataView dv = a.AsDataView("F");
            using NDArray back = dv.ToNDArray(dv.Schema["F"]);
            AssertShape(back, 3, 4);
        }

        [TestMethod]
        public void MultipleColumns_MaterializeIndependently()
        {
            var rows = new[]
            {
                new ScalarRow { A = 1f, B = 10f, Flag = true },
                new ScalarRow { A = 2f, B = 20f, Flag = false },
            };
            IDataView dv = Ml.Data.LoadFromEnumerable(rows);
            using NDArray a1 = dv.ToNDArray("A");
            using NDArray a2 = dv.ToNDArray("A");   // re-cursoring the same view
            using NDArray b = dv.ToNDArray("B");
            Assert.IsTrue(np.array_equal(a1, a2));
            CollectionAssert.AreEqual(new float[] { 10, 20 }, b.ToArray<float>());
        }

        private class ScalarRow
        {
            public float A { get; set; }
            public float B { get; set; }
            public bool Flag { get; set; }
        }

        private class VecRow
        {
            [VectorType(3)] public float[] F { get; set; }
        }

        private class UnsizedVecRow
        {
            [VectorType] public float[] F { get; set; }
        }

        private class TextRow
        {
            public string T { get; set; }
        }

        /// <summary>A row with one column of each ML.NET time type, for <see cref="TimeColumns_Throw_AndNameTheTicksRoute"/>.</summary>
        private class TimeRow
        {
            /// <summary>A <c>DateTime</c> column (ML.NET's <c>DateTimeDataViewType</c>).</summary>
            public DateTime When { get; set; }

            /// <summary>A <c>DateTimeOffset</c> column (ML.NET's <c>DateTimeOffsetDataViewType</c>).</summary>
            public DateTimeOffset At { get; set; }

            /// <summary>A <c>TimeSpan</c> column (ML.NET's <c>TimeSpanDataViewType</c>).</summary>
            public TimeSpan Took { get; set; }
        }
    }
}
