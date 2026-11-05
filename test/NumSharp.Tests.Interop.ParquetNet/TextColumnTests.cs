using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.ParquetNet;
using Parquet;
using Parquet.Schema;
using Parquet.Serialization;

namespace NumSharp.Tests.Interop.ParquetNet
{
    /// <summary>
    ///     Text columns and the ParquetNet bridge, end to end. The bridge reads numeric and boolean columns into NDArrays
    ///     and does not read text (UTF-8 strings, large strings, binary, dictionary-encoded strings) into NumSharp's text
    ///     dtypes: the file loaders (<see cref="ParquetFile"/>, <c>ReadParquetAsNDArraysAsync</c>) skip such a column and
    ///     the direct readers (<c>ReadColumnAsNDArrayAsync</c>, <c>ReadAsNDArrayAsync</c>) refuse it with the bridge's
    ///     text message — while every numeric column NEXT to the text still reads as numpy reads it.
    /// </summary>
    /// <remarks>
    ///     <para>Two writers: a file pyarrow wrote (the engine behind pandas' <c>to_parquet</c>; the committed fixture
    ///     <c>Fixtures/text_columns.parquet</c> from <c>test/oracle/gen_parquet_text_fixture.py</c>, whose numpy view of
    ///     every column is embedded below as the expected values), and files Parquet.Net's own serializer writes at test
    ///     time. The numpy reference is pyarrow's <c>table.column(name).to_numpy()</c>: the dtype and the values (a
    ///     <c>-0.0</c> keeps its sign).</para>
    ///     <para>A program that needs the text reads it with Parquet.Net and builds the text array itself:
    ///     <see cref="TextRoute_ParquetNetStrings_IntoNumSharpText_AreNumpysView"/> holds that route to numpy's view of the
    ///     same column (<c>np.array(list)</c>'s width, an NA for a null).</para>
    /// </remarks>
    [TestClass]
    public class TextColumnTests
    {
        /// <summary>The pyarrow fixture's numeric and boolean columns — exactly what the loaders take — in schema order.</summary>
        private static readonly string[] NumericColumns = { "id", "price", "ratio", "small", "wide", "flag" };

        /// <summary>The fixture's text-like columns: UTF-8, large UTF-8, binary and a dictionary-encoded string.</summary>
        private static readonly string[] TextLikeColumns = { "name", "name_large", "blob", "category" };

        /// <summary>The bridge's refusal of a text column, as its direct readers word it.</summary>
        private const string TextRefusal = "not text into NumSharp's text dtypes (U, S, StringDType)";

        /// <summary>The absolute path of the committed pyarrow fixture.</summary>
        /// <returns>The path.</returns>
        /// <exception cref="FileNotFoundException">The fixture was not copied beside the test binaries.</exception>
        private static string Fixture()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "text_columns.parquet");
            if (!File.Exists(path))
                throw new FileNotFoundException("run test/oracle/gen_parquet_text_fixture.py and rebuild", path);
            return path;
        }

        /// <summary>
        ///     The loaders over pyarrow's file take exactly the numeric and boolean columns, each equal to pyarrow's numpy
        ///     view (dtype and values; the float32 column's values are float32 values; <c>-0.0</c> keeps its sign), and
        ///     skip every text-like and list column: <see cref="ParquetFile.Columns"/> omits them, the indexer refuses them
        ///     as not loadable (naming what is), <c>TryGetColumn</c> answers false, and <c>ReadParquetAsNDArraysAsync</c>
        ///     returns the numeric columns only.
        /// </summary>
        [TestMethod]
        public async Task PyarrowFile_NumericColumnsReadAsNumpy_TextColumnsSkippedByTheLoaders()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture()))
            {
                pf.RowCount.Should().Be(6);
                pf.Columns.Should().Equal(NumericColumns);

                // pyarrow -> numpy: id int64, price float64, ratio float32, small int8, wide uint16, flag bool.
                pf["id"].typecode.Should().Be(NPTypeCode.Int64);
                pf["id"].ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
                pf["price"].typecode.Should().Be(NPTypeCode.Double);
                double[] price = pf["price"].ToArray<double>();
                price.Should().Equal(0.5, 1.25, -3.0, 1e300, -0.0, 2.5);
                BitConverter.DoubleToInt64Bits(price[4]).Should().Be(BitConverter.DoubleToInt64Bits(-0.0), "-0.0 keeps its sign");
                pf["ratio"].typecode.Should().Be(NPTypeCode.Single);
                pf["ratio"].ToArray<float>().Should().Equal(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f);
                pf["small"].typecode.Should().Be(NPTypeCode.SByte);
                pf["small"].ToArray<sbyte>().Should().Equal((sbyte)-128, (sbyte)127, (sbyte)0, (sbyte)1, (sbyte)-1, (sbyte)5);
                pf["wide"].typecode.Should().Be(NPTypeCode.UInt16);
                pf["wide"].ToArray<ushort>().Should().Equal((ushort)0, (ushort)65535, (ushort)1, (ushort)2, (ushort)3, (ushort)4);
                pf["flag"].typecode.Should().Be(NPTypeCode.Boolean);
                pf["flag"].ToArray<bool>().Should().Equal(true, false, true, true, false, false);

                foreach (string text in TextLikeColumns.Append("tags"))
                {
                    pf.TryGetColumn(text, out NDArray none).Should().BeFalse(text);
                    none.Should().BeNull(text);
                    new Action(() => { NDArray _ = pf[text]; }).Should().Throw<ArgumentException>(text)
                        .WithMessage($"column '{text}' not found or not loadable. Available: {string.Join(", ", NumericColumns)}*");
                }
            }

            await using FileStream stream = File.OpenRead(Fixture());
            Dictionary<string, NDArray> all = await stream.ReadParquetAsNDArraysAsync();
            all.Keys.Should().BeEquivalentTo(NumericColumns, "the stream loader takes the numeric columns and skips the text");
        }

        /// <summary>
        ///     The direct readers over pyarrow's file refuse each text-like column — UTF-8, large UTF-8, BINARY bytes and a
        ///     DICTIONARY-encoded string — with <see cref="NotSupportedException"/> naming the column and the list of
        ///     dtypes the bridge reads, refuse the list-of-text column as a repeated field (in the file reader and in a row
        ///     group's), and still read a numeric column of the same file. Which refusal TEXT a text column gets is
        ///     <see cref="TextColumns_DirectReaders_GiveTheTextRefusal_UnderParquetNet6"/>'s open bug.
        /// </summary>
        [TestMethod]
        public async Task PyarrowFile_TextColumns_RefusedByTheDirectReaders()
        {
            await using FileStream stream = File.OpenRead(Fixture());
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
            foreach (string text in TextLikeColumns)
            {
                Func<Task> read = async () => await reader.ReadColumnAsNDArrayAsync(text);
                (await read.Should().ThrowAsync<NotSupportedException>(text))
                    .WithMessage($"Parquet column '{text}' has type *Supported: bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double, decimal.");
            }

            // A list column's data field is its nested leaf (tags.list.element), so the by-name lookup does not find "tags"
            // (ArgumentException); the field itself is refused as a repeated field.
            Func<Task> byName = async () => await reader.ReadColumnAsNDArrayAsync("tags");
            (await byName.Should().ThrowAsync<ArgumentException>()).WithMessage("column 'tags' not found*");
            DataField tags = reader.Schema.GetDataFields().Single(f => f.MaxRepetitionLevel > 0);
            Func<Task> list = async () => await reader.ReadColumnAsNDArrayAsync(tags);
            (await list.Should().ThrowAsync<NotSupportedException>()).WithMessage("*repeated/list field*");

            using NDArray id = await reader.ReadColumnAsNDArrayAsync("id");
            id.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);

            using ParquetRowGroupReader rowGroup = reader.OpenRowGroupReader(0);
            DataField name = reader.Schema.GetDataFields().Single(f => f.Name == "name");
            Func<Task> perGroup = async () => await rowGroup.ReadAsNDArrayAsync(name);
            (await perGroup.Should().ThrowAsync<NotSupportedException>()).WithMessage("*'name'*Supported: bool*");
        }

        /// <summary>
        ///     OPEN BUG (found by these tests, 2026-11-05): the bridge's TEXT refusal ("… this bridge does not read: it maps
        ///     numeric and boolean columns, not text into NumSharp's text dtypes (U, S, StringDType)") is keyed on
        ///     <c>field.ClrType == typeof(string) || field.ClrType == typeof(byte[])</c> (<c>ParquetConvert</c>'s
        ///     <c>RejectIfUnloadable</c>), but Parquet.Net 6 — the package floor, 6.1.0 — types a UTF-8 column (plain, large or
        ///     dictionary-encoded) as <c>ReadOnlyMemory&lt;char&gt;</c>, so a text column falls through to the generic "has no
        ///     NumSharp dtype" text, which does not say the column is text or that the bridge (not NumSharp) lacks the route.
        ///     Every text-like column of a pyarrow file and of a Parquet.Net file should get the text refusal; the fix is to
        ///     key the branch on Parquet.Net 6's text and binary CLR types as well.
        /// </summary>
        [TestMethod]
        [TestCategory("OpenBugs")]
        public async Task TextColumns_DirectReaders_GiveTheTextRefusal_UnderParquetNet6()
        {
            await using (FileStream stream = File.OpenRead(Fixture()))
            await using (ParquetReader reader = await ParquetReader.CreateAsync(stream))
            {
                foreach (string text in TextLikeColumns)
                {
                    Func<Task> read = async () => await reader.ReadColumnAsNDArrayAsync(text);
                    (await read.Should().ThrowAsync<NotSupportedException>(text)).WithMessage($"*'{text}'*{TextRefusal}*");
                }
            }

            string path = Path.Combine(Path.GetTempPath(), $"nsparquet_textmsg_{Guid.NewGuid():N}.parquet");
            try
            {
                await ParquetSerializer.SerializeAsync(new[] { new MixedRow { Id = 1, Value = 2, Name = "n", Blob = new byte[] { 1 } } }, path);
                await using FileStream stream = File.OpenRead(path);
                await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
                foreach (string text in new[] { "Name", "Blob" })
                {
                    Func<Task> read = async () => await reader.ReadColumnAsNDArrayAsync(text);
                    (await read.Should().ThrowAsync<NotSupportedException>(text)).WithMessage($"*'{text}'*{TextRefusal}*");
                }
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>A Parquet.Net POCO: text, bytes and a nullable text next to numbers.</summary>
        public class MixedRow
        {
            /// <summary>An int32 key.</summary>
            public int Id { get; set; }

            /// <summary>A float64 value.</summary>
            public double Value { get; set; }

            /// <summary>A UTF-8 text column.</summary>
            public string Name { get; set; }

            /// <summary>A binary column.</summary>
            public byte[] Blob { get; set; }
        }

        /// <summary>
        ///     A file Parquet.Net's own serializer writes: the loaders take <c>Id</c> and <c>Value</c> and skip the text and
        ///     binary columns, a projection that names a text column keeps the numeric one only (the text name is dropped,
        ///     not refused), and the direct reader refuses the text columns, naming them (the refusal's TEXT is
        ///     <see cref="TextColumns_DirectReaders_GiveTheTextRefusal_UnderParquetNet6"/>'s open bug).
        /// </summary>
        [TestMethod]
        public async Task ParquetNetFile_TextNextToNumbers_LoadersSkipDirectReadersRefuse()
        {
            string path = Path.Combine(Path.GetTempPath(), $"nsparquet_text_{Guid.NewGuid():N}.parquet");
            try
            {
                var rows = Enumerable.Range(0, 5).Select(i => new MixedRow
                {
                    Id = i, Value = i * 0.5, Name = i == 3 ? null : $"row {i} {(char)0xE9}", Blob = new[] { (byte)i, (byte)0xFF },
                }).ToList();
                await ParquetSerializer.SerializeAsync(rows, path);

                using (ParquetFile pf = ParquetFile.Load(path))
                {
                    pf.Columns.Should().Equal("Id", "Value");
                    pf["Id"].ToArray<int>().Should().Equal(0, 1, 2, 3, 4);
                    pf["Value"].ToArray<double>().Should().Equal(0.0, 0.5, 1.0, 1.5, 2.0);
                }

                using (ParquetFile projected = ParquetFile.Load(path, new ParquetLoadOptions { Columns = new[] { "Name", "Value" } }))
                    projected.Columns.Should().Equal(new[] { "Value" }, "a projected text column is skipped like any text column");

                await using FileStream stream = File.OpenRead(path);
                await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
                foreach (string text in new[] { "Name", "Blob" })
                {
                    Func<Task> read = async () => await reader.ReadColumnAsNDArrayAsync(text);
                    (await read.Should().ThrowAsync<NotSupportedException>(text)).WithMessage($"Parquet column '{text}' has type *Supported: bool*");
                }
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>The fixture's UTF-8 column, as Parquet.Net's serializer reads it (the property name is the column's).</summary>
        public class NameRow
        {
            /// <summary>The <c>name</c> column (null for a Parquet null).</summary>
            [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006", Justification = "The Parquet column is named 'name'.")]
            public string name { get; set; }
        }

        /// <summary>
        ///     The route for text the bridge does not read: Parquet.Net's own reader gives the strings (null for a null), and
        ///     NumSharp's text arrays built from them are numpy's view of the same column — pyarrow's
        ///     <c>['alpha', '', 'été', '\U0001F600', None, 'x\x00y']</c>: as <c>StringDType(na_object=None)</c> every value
        ///     with the null as NA; as <c>np.array(list)</c> of the non-null texts the width numpy gives (the longest in code
        ///     points: <c>alpha</c>'s 5; the combining sequence counts 4, the emoji 1).
        /// </summary>
        [TestMethod]
        public async Task TextRoute_ParquetNetStrings_IntoNumSharpText_AreNumpysView()
        {
            await using FileStream stream = File.OpenRead(Fixture());
            // Parquet.Net 6 returns the rows with the file's schema and metadata; the rows are its Data.
            DeserializationResult<NameRow> result = await ParquetSerializer.DeserializeAsync<NameRow>(stream);
            string[] texts = result.Data.Select(r => r.name).ToArray();

            string combining = "e" + (char)0x0301 + "t" + (char)0x00E9;
            string emoji = char.ConvertFromUtf32(0x1F600);
            texts.Should().Equal("alpha", "", combining, emoji, null, "x\0y");

            using NDArray withNa = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType((object)null));
            withNa.shape.Should().Equal(6);
            withNa.GetValue(4).Should().BeNull("a Parquet null is the descriptor's NA");
            for (int i = 0; i < texts.Length; i++)
                if (texts[i] != null)
                    withNa.GetValue(i).Should().Be(texts[i], $"element {i}");

            using NDArray fixedWidth = np.array(texts.Where(t => t != null).ToArray());
            fixedWidth.dtype.str.Should().Be("<U5", "numpy gives the five non-null texts <U5: alpha is the longest");
            NDArray.AsStringArray(fixedWidth).Should().Equal("alpha", "", combining, emoji, "x\0y");
        }
    }
}
