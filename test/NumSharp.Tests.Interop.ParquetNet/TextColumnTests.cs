using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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
    ///     Text columns and the ParquetNet bridge, end to end. The bridge reads a UTF-8 column (plain, large,
    ///     dictionary-encoded, or bytes the schema annotates as STRING / ENUM / JSON) into a <c>StringDType(na_object=None)</c>
    ///     array, a null as the NA, and a binary column (variable or fixed length) into an <c>S</c> array as wide as its
    ///     longest value, a null as <c>b""</c> — numpy's view of such a column is an object array of <c>str</c> /
    ///     <c>bytes</c> and <c>None</c>. A list of text is still refused (not rectangular), and every numeric column next to
    ///     the text reads as numpy reads it.
    /// </summary>
    /// <remarks>
    ///     <para>Two writers: a file pyarrow wrote (the engine behind pandas' <c>to_parquet</c>; the committed fixtures
    ///     <c>Fixtures/text_columns.parquet</c> and its three-row-group copy <c>text_columns_rg2.parquet</c> from
    ///     <c>test/oracle/gen_parquet_text_fixture.py</c>, whose numpy view of every column is embedded below), and files
    ///     Parquet.Net's own serializer writes at test time. The numpy reference is pyarrow's
    ///     <c>table.column(name).to_numpy()</c>; for bytes, numpy's <c>S</c> of the same values, which — like every
    ///     <c>S</c> array — drops trailing NUL bytes when an element is read (the buffer keeps them).</para>
    ///     <para>Which column is text is decided by both keys the bridge has (<c>ParquetConvert.TextKindOf</c>): Parquet.Net's
    ///     CLR type (<c>ReadOnlyMemory&lt;char&gt;</c> / <c>ReadOnlyMemory&lt;byte&gt;</c> in Parquet.Net 6) and the
    ///     column's Parquet schema (a JSON column is <c>ReadOnlyMemory&lt;byte&gt;</c> in Parquet.Net 6.1.0, yet text).</para>
    /// </remarks>
    [TestClass]
    public class TextColumnTests
    {
        /// <summary>The pyarrow fixture's numeric and boolean columns, in schema order.</summary>
        private static readonly string[] NumericColumns = { "id", "price", "ratio", "small", "wide", "flag" };

        /// <summary>Every column the loaders take, in schema order: the list-of-text <c>tags</c> is the one skipped.</summary>
        private static readonly string[] Loadable =
            { "id", "price", "ratio", "small", "wide", "flag", "name", "name_large", "blob", "category", "doc", "fixed", "code" };

        /// <summary>The text columns (StringDType) and the byte columns (S).</summary>
        private static readonly string[] TextColumns = { "name", "name_large", "category", "doc", "code" }, ByteColumns = { "blob", "fixed" };

        /// <summary>The combining sequence <c>e + U+0301 + t + é</c> (four code points) in the <c>name</c> column.</summary>
        private static readonly string Combining = "e" + (char)0x0301 + "t" + (char)0x00E9;

        /// <summary>U+1F600, one code point, two UTF-16 units.</summary>
        private static readonly string Emoji = char.ConvertFromUtf32(0x1F600);

        /// <summary>numpy's view of <c>name</c> / <c>name_large</c>: <c>['alpha', '', 'été', '\U0001f600', None, 'x\x00y']</c>.</summary>
        private static readonly string[] Name = { "alpha", "", Combining, Emoji, null, "x\0y" };

        /// <summary>numpy's view of the dictionary-encoded <c>category</c>.</summary>
        private static readonly string[] Category = { "cat", "dog", "cat", "bird", "dog", "cat" };

        /// <summary>numpy's view of the JSON column <c>doc</c> (pyarrow's json extension reads <c>str</c>).</summary>
        private static readonly string[] Doc = { "{\"a\": 1}", null, "[]", "\"\"", "{}", "null" };

        /// <summary>numpy's view of the REQUIRED text column <c>code</c> (a tab and a lone space kept).</summary>
        private static readonly string[] Code = { "", ((char)0xE9).ToString(), Emoji, "tab\t", " ", "END" };

        /// <summary><c>blob</c> as numpy's <c>|S3</c> reads it: <c>[b'\x00\x01', b'', b'\xff', b'abc', b'' (the null), b'z']</c>.</summary>
        private static readonly byte[][] Blob = { new byte[] { 0, 1 }, new byte[0], new byte[] { 0xFF }, "abc"u8.ToArray(), new byte[0], "z"u8.ToArray() };

        /// <summary>
        ///     <c>fixed</c> (binary(2)) as numpy's <c>|S2</c> reads it: <c>b'ab'</c>, <c>b'cd'</c>, <c>b''</c> (the null),
        ///     <c>b''</c> (<c>\x00\x00</c>, all trailing NULs), <c>b'z'</c> (<c>z\x00</c>), <c>b'yy'</c>.
        /// </summary>
        private static readonly byte[][] Fixed = { "ab"u8.ToArray(), "cd"u8.ToArray(), new byte[0], new byte[0], "z"u8.ToArray(), "yy"u8.ToArray() };

        /// <summary>The absolute path of a committed pyarrow fixture.</summary>
        /// <param name="name">The fixture's file name.</param>
        /// <returns>The path.</returns>
        /// <exception cref="FileNotFoundException">The fixture was not copied beside the test binaries.</exception>
        private static string Fixture(string name = "text_columns.parquet")
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", name);
            if (!File.Exists(path))
                throw new FileNotFoundException("run test/oracle/gen_parquet_text_fixture.py and rebuild", path);
            return path;
        }

        /// <summary>Asserts a text column is <c>StringDType(na_object=None)</c> holding <paramref name="expected"/> (null = the NA).</summary>
        /// <param name="nd">The column read.</param>
        /// <param name="expected">numpy's view: the texts, null for <c>None</c>.</param>
        /// <param name="because">The context named in a failure.</param>
        private static void AssertText(NDArray nd, string[] expected, string because)
        {
            nd.dtype.ToString().Should().Be("StringDType(na_object=None)", because);
            nd.shape.Should().Equal(new long[] { expected.Length }, because);
            for (int i = 0; i < expected.Length; i++)
                nd.GetValue(i).Should().Be(expected[i], $"{because}: element {i}");
        }

        /// <summary>Asserts a byte column is an <c>S</c> array of <paramref name="dtype"/> whose element reads are <paramref name="expected"/>.</summary>
        /// <param name="nd">The column read.</param>
        /// <param name="dtype">numpy's <c>dtype.str</c> (<c>|S3</c> for the longest value of three bytes).</param>
        /// <param name="expected">numpy's element reads (trailing NULs dropped, <c>b""</c> for a null).</param>
        /// <param name="because">The context named in a failure.</param>
        private static void AssertBytes(NDArray nd, string dtype, byte[][] expected, string because)
        {
            nd.dtype.str.Should().Be(dtype, because);
            nd.shape.Should().Equal(new long[] { expected.Length }, because);
            for (int i = 0; i < expected.Length; i++)
                ((byte[])nd.GetValue(i)).Should().Equal(expected[i], $"{because}: element {i}");
        }

        /// <summary>Asserts every text and byte column of the pyarrow fixture, read through <paramref name="column"/>, is numpy's view.</summary>
        /// <param name="column">Reads a column by name.</param>
        /// <param name="because">The context named in a failure.</param>
        private static void AssertTextColumns(Func<string, NDArray> column, string because)
        {
            AssertText(column("name"), Name, $"{because}: name");
            AssertText(column("name_large"), Name, $"{because}: name_large");
            AssertText(column("category"), Category, $"{because}: category (dictionary-encoded)");
            AssertText(column("doc"), Doc, $"{because}: doc (JSON bytes, text by the schema key)");
            AssertText(column("code"), Code, $"{because}: code (required)");
            AssertBytes(column("blob"), "|S3", Blob, $"{because}: blob");
            AssertBytes(column("fixed"), "|S2", Fixed, $"{because}: fixed");
        }

        /// <summary>
        ///     The loaders over pyarrow's file take every flat column in schema order — numbers as numpy's view (dtype and
        ///     values; <c>-0.0</c> keeps its sign), text as <c>StringDType(na_object=None)</c> with the null as the NA, bytes
        ///     as <c>S</c> — and skip only the list of text: <see cref="ParquetFile.Columns"/> omits it, the indexer refuses it
        ///     naming what is loadable, <c>TryGetColumn</c> answers false, and <c>ReadParquetAsNDArraysAsync</c> returns the
        ///     same columns with the same values.
        /// </summary>
        /// <returns>A task that completes when the extension's columns have been checked.</returns>
        [TestMethod]
        public async Task PyarrowFile_EveryFlatColumn_ReadsAsNumpysView_TheListIsSkipped()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture()))
            {
                pf.RowCount.Should().Be(6);
                pf.Columns.Should().Equal(Loadable);

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

                AssertTextColumns(name => pf[name], "ParquetFile");
                pf["fixed"].tobytes().Should().Equal(new byte[] { 0x61, 0x62, 0x63, 0x64, 0, 0, 0, 0, 0x7A, 0, 0x79, 0x79 },
                    "the S buffer keeps the trailing NULs an element read drops (the null is b\"\" padded to two bytes)");

                pf.TryGetColumn("tags", out NDArray none).Should().BeFalse();
                none.Should().BeNull();
                new Action(() => { NDArray _ = pf["tags"]; }).Should().Throw<ArgumentException>()
                    .WithMessage($"column 'tags' not found or not loadable. Available: {string.Join(", ", Loadable)}*");
            }

            await using FileStream stream = File.OpenRead(Fixture());
            Dictionary<string, NDArray> all = await stream.ReadParquetAsNDArraysAsync();
            all.Keys.Should().Equal(Loadable, "the stream loader takes the same columns in schema order");
            AssertTextColumns(name => all[name], "ReadParquetAsNDArraysAsync");
        }

        /// <summary>
        ///     The direct readers over pyarrow's file read the text and byte columns as the loaders do — by name, by field and
        ///     per row group — and refuse the list of text: by name it is not found (its data field is the nested leaf
        ///     <c>tags.list.element</c>), as a field it is a repeated field. The schema key is visible here: the JSON column's
        ///     Parquet.Net type is <c>ReadOnlyMemory&lt;byte&gt;</c>, yet it reads as text, and the fixed-width binary column
        ///     (<c>FIXED_LEN_BYTE_ARRAY</c>) reads as <c>S</c>.
        /// </summary>
        /// <returns>A task that completes when every reader has been tried.</returns>
        [TestMethod]
        public async Task PyarrowFile_DirectReaders_ReadText_RefuseTheList()
        {
            await using FileStream stream = File.OpenRead(Fixture());
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
            DataField[] fields = reader.Schema.GetDataFields();
            fields.Single(f => f.Name == "doc").ClrType.Should().Be(typeof(ReadOnlyMemory<byte>), "Parquet.Net 6.1.0 hands JSON over as bytes");
            fields.Single(f => f.Name == "name").ClrType.Should().Be(typeof(ReadOnlyMemory<char>));

            var byName = new Dictionary<string, NDArray>();
            var byField = new Dictionary<string, NDArray>();
            var perGroup = new Dictionary<string, NDArray>();
            using (ParquetRowGroupReader rowGroup = reader.OpenRowGroupReader(0))
            {
                foreach (string name in TextColumns.Concat(ByteColumns))
                {
                    DataField field = fields.Single(f => f.Name == name);
                    byName[name] = await reader.ReadColumnAsNDArrayAsync(name);
                    byField[name] = await reader.ReadColumnAsNDArrayAsync(field);
                    perGroup[name] = await rowGroup.ReadAsNDArrayAsync(field);
                }
            }

            AssertTextColumns(name => byName[name], "ReadColumnAsNDArrayAsync(name)");
            AssertTextColumns(name => byField[name], "ReadColumnAsNDArrayAsync(field)");
            AssertTextColumns(name => perGroup[name], "ReadAsNDArrayAsync (one row group)");

            Action listByName = () => reader.ReadColumnAsNDArrayAsync("tags");
            listByName.Should().Throw<ArgumentException>().WithMessage("column 'tags' not found*");
            DataField tags = fields.Single(f => f.MaxRepetitionLevel > 0);
            Action list = () => reader.ReadColumnAsNDArrayAsync(tags);
            list.Should().Throw<NotSupportedException>().WithMessage("*repeated/list field*");
        }

        /// <summary>
        ///     The same table in three row groups of two rows: every text and byte column streams through both faces — the
        ///     NumSharp face (<see cref="ParquetFile.RowGroups"/>) and Parquet.Net's (<c>ParquetRowGroupReader.ReadAsNDArrayAsync</c>)
        ///     — and the groups' VALUES, in order, are the whole column's. Each group's <c>S</c> is as wide as its own longest
        ///     value (numpy's <c>np.array</c> per group), so groups are compared by value, not by buffer.
        /// </summary>
        /// <returns>A task that completes when both faces have been compared.</returns>
        [TestMethod]
        public async Task RowGroups_TextAndBytes_StreamAsTheWholeColumnsValues_BothFaces()
        {
            static object[] Values(NDArray nd) => Enumerable.Range(0, (int)nd.size).Select(i => nd.GetValue(i)).ToArray();

            using ParquetFile pf = ParquetFile.Load(Fixture("text_columns_rg2.parquet"));
            pf.RowGroupCount.Should().Be(3);
            AssertTextColumns(name => pf[name], "the whole rg2 file");
            var streamed = TextColumns.Concat(ByteColumns).ToDictionary(n => n, _ => new List<object>());
            foreach (ParquetRowGroup group in pf.RowGroups)
            {
                using (group)
                {
                    group.RowCount.Should().Be(2);
                    foreach (string name in streamed.Keys)
                        streamed[name].AddRange(Values(group.Column(name)));
                }
            }

            foreach (string name in streamed.Keys)
                streamed[name].Should().BeEquivalentTo(Values(pf[name]), o => o.WithStrictOrdering(), $"{name}: the groups' values are the whole column's");

            await using FileStream stream = File.OpenRead(Fixture("text_columns_rg2.parquet"));
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
            DataField blob = reader.Schema.GetDataFields().Single(f => f.Name == "blob");
            var widths = new List<string>();
            for (int g = 0; g < reader.RowGroupCount; g++)
            {
                using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
                widths.Add((await rg.ReadAsNDArrayAsync(blob)).dtype.str);
            }

            widths.Should().Equal(new[] { "|S2", "|S3", "|S1" }, "each group's S is as wide as its own longest value");
        }

        /// <summary>
        ///     The null policy over text and bytes: by default a text null is the NA and a byte null <c>b""</c>; a file-level
        ///     fill converts into each column as NumPy's <c>a[i] = fill</c> does (text stays text; NaN is the text
        ///     <c>'nan'</c> / the bytes <c>b'nan'</c>, the <c>S</c> widening to hold it), only where a null is; a per-call fill
        ///     is fresh and leaves the cached default; under <see cref="NullHandling.Raise"/> a column with a null is refused
        ///     while the required and the null-free text columns read.
        /// </summary>
        [TestMethod]
        public void TextColumns_NullPolicy_Fill_PerCall_Raise()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture()))
            {
                pf["name"].GetValue(4).Should().BeNull("a text null is the NA by default");
                ((byte[])pf["blob"].GetValue(4)).Should().BeEmpty("a byte null is b\"\" by default (S has no NA)");
                NDArray dashed = pf.Column("name", "-");
                dashed.GetValue(4).Should().Be("-", "a per-call fill");
                pf["name"].GetValue(4).Should().BeNull("the cached default keeps its NA");
            }

            using (ParquetFile filled = ParquetFile.Load(Fixture(), new ParquetLoadOptions { NullFill = "?" }))
            {
                AssertText(filled["name"], Name.Select(t => t ?? "?").ToArray(), "NullFill '?'");
                AssertText(filled["doc"], Doc.Select(t => t ?? "?").ToArray(), "NullFill '?' into the JSON column");
                AssertBytes(filled["blob"], "|S3", Blob.Select((b, i) => i == 4 ? "?"u8.ToArray() : b).ToArray(), "NullFill '?' into bytes");
                AssertText(filled["code"], Code, "a required column has no null to fill");
            }

            using (ParquetFile nan = ParquetFile.Load(Fixture(), new ParquetLoadOptions { NullFill = double.NaN }))
            {
                nan["name"].GetValue(4).Should().Be("nan", "numpy: np.array([..., np.nan], dtype=StringDType(na_object=None)) stores 'nan'");
                ((byte[])nan["blob"].GetValue(4)).Should().Equal("nan"u8.ToArray(), "numpy stores b'nan' into S");
                ((byte[])nan["fixed"].GetValue(2)).Should().Equal("nan"u8.ToArray(), "the fixed column's S widens to hold b'nan'");
                nan["fixed"].dtype.str.Should().Be("|S3");
                nan["price"].ToArray<double>().Should().HaveCount(6, "a numeric column without nulls reads under the NaN fill");
            }

            using (ParquetFile raise = ParquetFile.Load(Fixture(), new ParquetLoadOptions { Nulls = NullHandling.Raise }))
            {
                foreach (string name in new[] { "name", "name_large", "doc", "blob", "fixed" })
                {
                    Action read = () => { NDArray _ = raise[name]; };
                    read.Should().Throw<InvalidOperationException>(name)
                        .WithMessage($"Parquet column '{name}' contains null values but NullHandling.Raise was requested.");
                }

                AssertText(raise["code"], Code, "Raise reads the required text column");
                AssertText(raise["category"], Category, "Raise reads an optional text column without nulls");
                Action perCall = () => raise.Column("name", "-");
                perCall.Should().Throw<InvalidOperationException>("a per-call fill keeps the file's Raise policy");
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
        ///     A file Parquet.Net's own serializer writes: the loaders take all four columns in schema order — the text as
        ///     <c>StringDType(na_object=None)</c> with the null row as the NA, the bytes as <c>|S2</c> — a projection keeps the
        ///     text column it names (in schema order), and the direct reader reads the same values.
        /// </summary>
        /// <returns>A task that completes when the direct reader's columns have been checked.</returns>
        [TestMethod]
        public async Task ParquetNetFile_TextNextToNumbers_LoadersAndDirectReadersReadIt()
        {
            string path = Path.Combine(Path.GetTempPath(), $"nsparquet_text_{Guid.NewGuid():N}.parquet");
            try
            {
                var rows = Enumerable.Range(0, 5).Select(i => new MixedRow
                {
                    Id = i, Value = i * 0.5, Name = i == 3 ? null : $"row {i} {(char)0xE9}", Blob = new[] { (byte)i, (byte)0xFF },
                }).ToList();
                await ParquetSerializer.SerializeAsync(rows, path);
                string[] names = rows.Select(r => r.Name).ToArray();
                byte[][] blobs = rows.Select(r => r.Blob).ToArray();

                using (ParquetFile pf = ParquetFile.Load(path))
                {
                    pf.Columns.Should().Equal("Id", "Value", "Name", "Blob");
                    pf["Id"].ToArray<int>().Should().Equal(0, 1, 2, 3, 4);
                    pf["Value"].ToArray<double>().Should().Equal(0.0, 0.5, 1.0, 1.5, 2.0);
                    AssertText(pf["Name"], names, "Name");
                    AssertBytes(pf["Blob"], "|S2", blobs, "Blob");
                }

                using (ParquetFile projected = ParquetFile.Load(path, new ParquetLoadOptions { Columns = new[] { "Name", "Value" } }))
                    projected.Columns.Should().Equal(new[] { "Value", "Name" }, "a projection keeps the text column, in schema order");

                await using FileStream stream = File.OpenRead(path);
                await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
                AssertText(await reader.ReadColumnAsNDArrayAsync("Name"), names, "the direct reader");
                AssertBytes(await reader.ReadColumnAsNDArrayAsync("Blob"), "|S2", blobs, "the direct reader");
            }
            finally
            {
                File.Delete(path);
            }
        }

        /// <summary>
        ///     The schema key's refusal: a column the Parquet schema annotates as UTF-8 text but that Parquet.Net would decode
        ///     to a CLR type the bridge does not read is refused AS TEXT (NumSharp has text dtypes; only that decode is
        ///     missing), not as a type with "no NumSharp dtype"; the same field without the annotation gets the generic
        ///     refusal. Parquet.Net 6.1.0 never builds such a field (every UTF-8 column is <c>ReadOnlyMemory&lt;char&gt;</c> or,
        ///     for JSON, bytes), so the test sets the field's schema element through its non-public setter — the guard is for
        ///     other Parquet.Net versions.
        /// </summary>
        /// <returns>A task that completes when both refusals have been observed.</returns>
        [TestMethod]
        public async Task SchemaKey_Utf8ColumnTheBridgeCannotDecode_IsRefusedAsText()
        {
            await using FileStream stream = File.OpenRead(Fixture());
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream);

            var annotated = new DataField("when", typeof(Guid));
            PropertyInfo schemaElement = typeof(DataField).GetProperty(nameof(DataField.SchemaElement))!;
            schemaElement.GetSetMethod(nonPublic: true)!.Invoke(annotated, new object[]
            {
                new Parquet.Meta.SchemaElement { Type = Parquet.Meta.Type.BYTE_ARRAY, ConvertedType = Parquet.Meta.ConvertedType.UTF8 },
            });
            Action text = () => reader.ReadColumnAsNDArrayAsync(annotated);
            text.Should().Throw<NotSupportedException>().WithMessage(
                "Parquet column 'when' holds UTF-8 text, which Parquet.Net decodes as System.Guid; this bridge reads text Parquet.Net " +
                "decodes as ReadOnlyMemory<char> or string, or as UTF-8 bytes (ReadOnlyMemory<byte> or byte[]).");

            Action plain = () => reader.ReadColumnAsNDArrayAsync(new DataField("when", typeof(Guid)));
            plain.Should().Throw<NotSupportedException>().WithMessage(
                "Parquet column 'when' has type System.Guid, which has no NumSharp dtype. Supported: bool, byte, sbyte, short, ushort, " +
                "int, uint, long, ulong, float, double, decimal, text and binary.");
        }

        /// <summary>The fixture's UTF-8 column, as Parquet.Net's serializer reads it (the property name is the column's).</summary>
        public class NameRow
        {
            /// <summary>The <c>name</c> column (null for a Parquet null).</summary>
            [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006", Justification = "The Parquet column is named 'name'.")]
            public string name { get; set; }
        }

        /// <summary>
        ///     The bridge's read against the hand route a program would otherwise take: Parquet.Net's own reader gives the
        ///     strings (null for a null), and NumSharp's text arrays built from them are numpy's view of the same column —
        ///     pyarrow's <c>['alpha', '', 'été', '\U0001F600', None, 'x\x00y']</c>: as <c>StringDType(na_object=None)</c>
        ///     every value with the null as NA (what the bridge reads), as <c>np.array(list)</c> of the non-null texts the width
        ///     numpy gives (the longest in code points: <c>alpha</c>'s 5; the combining sequence counts 4, the emoji 1).
        /// </summary>
        /// <returns>A task that completes when the route has been compared.</returns>
        [TestMethod]
        public async Task TextRoute_ParquetNetStrings_IntoNumSharpText_AreTheBridgesRead()
        {
            await using FileStream stream = File.OpenRead(Fixture());
            // Parquet.Net 6 returns the rows with the file's schema and metadata; the rows are its Data.
            DeserializationResult<NameRow> result = await ParquetSerializer.DeserializeAsync<NameRow>(stream);
            string[] texts = result.Data.Select(r => r.name).ToArray();
            texts.Should().Equal(Name);

            using NDArray withNa = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType((object)null));
            AssertText(withNa, Name, "the hand route");
            using (ParquetFile pf = ParquetFile.Load(Fixture()))
                for (int i = 0; i < texts.Length; i++)
                    pf["name"].GetValue(i).Should().Be(withNa.GetValue(i), $"element {i}: the bridge reads what the route builds");

            using NDArray fixedWidth = np.array(texts.Where(t => t != null).ToArray());
            fixedWidth.dtype.str.Should().Be("<U5", "numpy gives the five non-null texts <U5: alpha is the longest");
            NDArray.AsStringArray(fixedWidth).Should().Equal("alpha", "", Combining, Emoji, "x\0y");
        }
    }
}
