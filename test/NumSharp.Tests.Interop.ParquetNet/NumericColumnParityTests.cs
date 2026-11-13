using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.ParquetNet;
using Parquet;
using Parquet.Schema;

namespace NumSharp.Tests.Interop.ParquetNet
{
    /// <summary>
    ///     The numeric neighbours of the text columns, end to end: Parquet files pyarrow wrote (the engine behind pandas'
    ///     <c>to_parquet</c>), read through the bridge, every column held to pyarrow's own <c>to_numpy()</c> view — the dtype
    ///     and the BITS (<c>-0.0</c>'s sign, NaN, the smallest subnormals, the largest finite values, every integer width's
    ///     minimum and maximum, the decimal's scale) — in every writer configuration and across row-group boundaries.
    /// </summary>
    /// <remarks>
    ///     <para>The committed fixtures come from <c>test/oracle/gen_parquet_numeric_fixture.py</c>: one ten-row table in three
    ///     row groups (4 + 4 + 2) under four writer configurations — pyarrow's defaults (dictionary pages, snappy), PLAIN +
    ///     zstd, DELTA_BINARY_PACKED / BYTE_STREAM_SPLIT + gzip with the decimal stored as an INT64, and data page v2 — plus a
    ///     zero-row copy, a file of decimals at and past .NET decimal's limits, and a float16 file. The expected values below
    ///     are the generator's printed numpy view.</para>
    ///     <para>The null policy is the bridge's own, and differs from numpy's view on purpose: pyarrow / pandas widen a
    ///     nullable integer to float64 with NaN and a nullable boolean to an object array with None, where the bridge keeps the
    ///     column's dtype and fills the nulls (<c>default(T)</c>, a custom fill, or a refusal under
    ///     <see cref="NullHandling.Raise"/>). A fill value is held to numpy's <c>a[i] = fill</c> casting (NumSharp's
    ///     <c>PyArray_Pack</c> port), converted only where a null is.</para>
    ///     <para>Pinned in this class: the limits of Parquet.Net 6.1.0 or of the 96-bit decimal (<c>[Misaligned]</c> — data
    ///     page v2 booleans, DELTA_BINARY_PACKED int8 / uint8 / decimal, float16 schemas, decimals past 28-29 digits, the
    ///     decimal's trailing zeros).</para>
    /// </remarks>
    [TestClass]
    public class NumericColumnParityTests
    {
        /// <summary>pyarrow's default writer configuration: dictionary pages, snappy, data page v1.</summary>
        private const string DefaultFile = "numeric_columns.parquet";

        /// <summary>Data page v2: the booleans are RLE-encoded, which Parquet.Net 6.1.0 cannot decode.</summary>
        private const string PageV2File = "numeric_columns_v2.parquet";

        /// <summary>
        ///     DELTA_BINARY_PACKED integers (and the decimal stored as an INT64), BYTE_STREAM_SPLIT floats, gzip: Parquet.Net
        ///     6.1.0 decodes the delta encoding into int16..uint64 but not into int8, uint8 or decimal.
        /// </summary>
        private const string DeltaFile = "numeric_columns_delta_gzip.parquet";

        /// <summary>The loadable columns in schema order: every numeric / boolean / decimal column, not the two temporal ones.</summary>
        private static readonly string[] Loadable =
        {
            "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64", "flag", "dec",
            "opt_i32", "maybe_i64", "maybe_f64", "maybe_u8", "maybe_flag", "all_null",
        };

        /// <summary>The columns that hold nulls; numpy widens each (float64 + NaN, or object + None), the bridge fills.</summary>
        private static readonly string[] WithNulls = { "maybe_i64", "maybe_f64", "maybe_u8", "maybe_flag", "all_null" };

        /// <summary>The float32 column as pyarrow wrote it: 0.1's float32, -0.0, NaN, ±inf, the smallest subnormal, the largest finite.</summary>
        private static readonly float[] F32 =
            { 0.1f, -0.0f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, float.Epsilon, float.MaxValue, 1f, 2.5f, -7.25f };

        /// <summary>The float64 column as pyarrow wrote it: 0.1, -0.0, NaN, ±inf, 5e-324, the largest finite.</summary>
        private static readonly double[] F64 =
            { 0.1, -0.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.Epsilon, double.MaxValue, 1, 2.5, -7.25 };

        /// <summary>The required boolean column.</summary>
        private static readonly bool[] Flag = { true, false, true, true, false, false, true, false, true, false };

        /// <summary>The decimal128(10, 2) column as numpy's <c>str(Decimal)</c> prints it — the scale (two digits) included.</summary>
        private static readonly string[] Decimals =
            { "1.25", "-3.50", "0.00", "99999999.99", "-0.01", "7.00", "8.10", "9.99", "10.00", "-12345678.90" };

        /// <summary>The nullable int64 column (numpy's view: float64 with NaN at the nulls).</summary>
        private static readonly long?[] MaybeI64 = { 1, null, 3, null, 5, 6, null, 8, 9, null };

        /// <summary>The nullable uint8 column (numpy's view: float64 with NaN at the nulls).</summary>
        private static readonly byte?[] MaybeU8 = { 1, null, 255, 0, null, 5, 6, 7, null, 9 };

        /// <summary>The nullable boolean column (numpy's view: an object array with None at the nulls).</summary>
        private static readonly bool?[] MaybeFlag = { true, null, false, true, null, false, true, null, false, true };

        /// <summary>The nullable float64 column (numpy's view: NaN at the nulls; the written -0.0 keeps its sign).</summary>
        private static readonly double?[] MaybeF64 = { 1.5, null, -0.0, null, 2.5, null, 3.5, 4.5, null, 5.5 };

        /// <summary>The bridge's refusal of a column whose CLR type has no NumSharp dtype (the timestamp and the date).</summary>
        private const string NoDtype =
            "Parquet column '{0}' has type System.DateTime, which has no NumSharp dtype. " +
            "Supported: bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double, decimal, text and binary.";

        /// <summary>The absolute path of a committed fixture.</summary>
        /// <param name="name">The fixture's file name under <c>Fixtures/</c>.</param>
        /// <returns>The path.</returns>
        /// <exception cref="FileNotFoundException">The fixture was not copied beside the test binaries.</exception>
        private static string Fixture(string name)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", name);
            if (!File.Exists(path))
                throw new FileNotFoundException("run test/oracle/gen_parquet_numeric_fixture.py and rebuild", path);
            return path;
        }

        /// <summary>A nullable column with its nulls replaced — what the bridge's fill policy produces.</summary>
        /// <typeparam name="T">The column's element type.</typeparam>
        /// <param name="column">The column, null at the nulls.</param>
        /// <param name="fill">The value each null becomes.</param>
        /// <returns>The filled column.</returns>
        private static T[] Filled<T>(T?[] column, T fill) where T : struct => column.Select(v => v ?? fill).ToArray();

        /// <summary>
        ///     Asserts two float64 sequences are bit-identical, except that a NaN matches any NaN (C#'s <c>double.NaN</c> is the
        ///     NEGATIVE quiet NaN and Python's <c>float('nan')</c> the positive one, so a NaN's sign is not the value under test).
        /// </summary>
        /// <param name="actual">The values read.</param>
        /// <param name="expected">numpy's values.</param>
        /// <param name="because">The context named in a failure.</param>
        private static void ShouldBeBits(double[] actual, double[] expected, string because)
        {
            actual.Length.Should().Be(expected.Length, because);
            for (int i = 0; i < expected.Length; i++)
            {
                if (double.IsNaN(expected[i]))
                    double.IsNaN(actual[i]).Should().BeTrue($"{because}: element {i} is NaN");
                else
                    BitConverter.DoubleToInt64Bits(actual[i]).Should().Be(BitConverter.DoubleToInt64Bits(expected[i]), $"{because}: element {i}");
            }
        }

        /// <summary>Asserts two float32 sequences are bit-identical, a NaN matching any NaN (see the float64 overload).</summary>
        /// <param name="actual">The values read.</param>
        /// <param name="expected">numpy's values.</param>
        /// <param name="because">The context named in a failure.</param>
        private static void ShouldBeBits(float[] actual, float[] expected, string because)
        {
            actual.Length.Should().Be(expected.Length, because);
            for (int i = 0; i < expected.Length; i++)
            {
                if (float.IsNaN(expected[i]))
                    float.IsNaN(actual[i]).Should().BeTrue($"{because}: element {i} is NaN");
                else
                    BitConverter.SingleToInt32Bits(actual[i]).Should().Be(BitConverter.SingleToInt32Bits(expected[i]), $"{because}: element {i}");
            }
        }

        /// <summary>
        ///     Asserts every non-boolean column WITHOUT nulls is numpy's view: each integer width at its extremes in its own
        ///     dtype, both float widths bit for bit, the decimal's values, and the optional int32 that holds no null.
        /// </summary>
        /// <remarks>
        ///     The decimal is compared by VALUE (<c>-3.5m == -3.50m</c>): Parquet.Net's decode drops the trailing zeros of the
        ///     column's scale, which <see cref="Decimal_TrailingZerosOfTheScale_DropInParquetNetsDecode"/> pins on its own.
        /// </remarks>
        /// <param name="column">Reads a column by name (the file's indexer, or the dictionary the extension returns).</param>
        /// <param name="because">The context named in a failure (the fixture and the route).</param>
        /// <param name="except">Columns this caller pins separately (a configuration Parquet.Net cannot decode them in).</param>
        private static void AssertNumericColumnsAreNumpysView(Func<string, NDArray> column, string because, params string[] except)
        {
            NDArray Typed(string name, NPTypeCode dtype)
            {
                NDArray nd = column(name);
                nd.typecode.Should().Be(dtype, $"{because}: {name}'s dtype");
                nd.shape.Should().Equal(new long[] { 10 }, $"{because}: {name} is one row per element");
                return nd;
            }

            bool Wants(string name) => Array.IndexOf(except, name) < 0;

            if (Wants("i8"))
                Typed("i8", NPTypeCode.SByte).ToArray<sbyte>().Should().Equal(new sbyte[] { -128, 127, 0, 1, -1, 5, 7, 8, 9, 10 }, because);
            if (Wants("i16"))
                Typed("i16", NPTypeCode.Int16).ToArray<short>().Should().Equal(new short[] { -32768, 32767, 0, 1, -1, 5, 7, 8, 9, 10 }, because);
            if (Wants("i32"))
                Typed("i32", NPTypeCode.Int32).ToArray<int>().Should().Equal(new[] { int.MinValue, int.MaxValue, 0, 1, -1, 5, 7, 8, 9, 10 }, because);
            if (Wants("i64"))
                Typed("i64", NPTypeCode.Int64).ToArray<long>().Should().Equal(new[] { long.MinValue, long.MaxValue, 0L, 1L, -1L, 5L, 7L, 8L, 9L, 10L }, because);
            if (Wants("u8"))
                Typed("u8", NPTypeCode.Byte).ToArray<byte>().Should().Equal(new byte[] { 0, 255, 0, 1, 2, 5, 7, 8, 9, 10 }, because);
            if (Wants("u16"))
                Typed("u16", NPTypeCode.UInt16).ToArray<ushort>().Should().Equal(new ushort[] { 0, 65535, 0, 1, 2, 5, 7, 8, 9, 10 }, because);
            if (Wants("u32"))
                Typed("u32", NPTypeCode.UInt32).ToArray<uint>().Should().Equal(new[] { 0u, uint.MaxValue, 0u, 1u, 2u, 5u, 7u, 8u, 9u, 10u }, because);
            if (Wants("u64"))
                Typed("u64", NPTypeCode.UInt64).ToArray<ulong>().Should().Equal(new[] { 0ul, ulong.MaxValue, 0ul, 1ul, 2ul, 5ul, 7ul, 8ul, 9ul, 10ul }, because);
            if (Wants("f32"))
                ShouldBeBits(Typed("f32", NPTypeCode.Single).ToArray<float>(), F32, $"{because}: f32");
            if (Wants("f64"))
                ShouldBeBits(Typed("f64", NPTypeCode.Double).ToArray<double>(), F64, $"{because}: f64");
            if (Wants("dec"))
                Typed("dec", NPTypeCode.Decimal).ToArray<decimal>()
                    .Should().Equal(Decimals.Select(s => decimal.Parse(s, CultureInfo.InvariantCulture)), $"{because}: dec's values");
            if (Wants("opt_i32"))
                Typed("opt_i32", NPTypeCode.Int32).ToArray<int>().Should().Equal(Enumerable.Range(100, 10), $"{because}: an optional column without nulls");
        }

        /// <summary>
        ///     Asserts the nullable columns read in their own dtype with each null filled by <c>default(T)</c> — the bridge's
        ///     default policy (numpy's view widens the integers to float64 + NaN and the boolean to object + None).
        /// </summary>
        /// <param name="column">Reads a column by name.</param>
        /// <param name="because">The context named in a failure.</param>
        /// <param name="except">Columns this caller pins separately (a configuration Parquet.Net cannot decode them in).</param>
        private static void AssertNullableColumnsFillDefault(Func<string, NDArray> column, string because, params string[] except)
        {
            bool Wants(string name) => Array.IndexOf(except, name) < 0;

            if (Wants("maybe_i64"))
            {
                column("maybe_i64").typecode.Should().Be(NPTypeCode.Int64, $"{because}: maybe_i64 keeps int64 (numpy: float64 + NaN)");
                column("maybe_i64").ToArray<long>().Should().Equal(Filled(MaybeI64, 0L), because);
            }

            if (Wants("maybe_u8"))
            {
                column("maybe_u8").typecode.Should().Be(NPTypeCode.Byte, $"{because}: maybe_u8 keeps uint8 (numpy: float64 + NaN)");
                column("maybe_u8").ToArray<byte>().Should().Equal(Filled(MaybeU8, (byte)0), because);
            }

            if (Wants("maybe_flag"))
            {
                column("maybe_flag").typecode.Should().Be(NPTypeCode.Boolean, $"{because}: maybe_flag keeps bool (numpy: object + None)");
                column("maybe_flag").ToArray<bool>().Should().Equal(Filled(MaybeFlag, false), because);
            }

            if (Wants("maybe_f64"))
            {
                column("maybe_f64").typecode.Should().Be(NPTypeCode.Double, because);
                ShouldBeBits(column("maybe_f64").ToArray<double>(), Filled(MaybeF64, 0.0), $"{because}: maybe_f64");
            }

            if (Wants("all_null"))
            {
                column("all_null").typecode.Should().Be(NPTypeCode.Double, because);
                ShouldBeBits(column("all_null").ToArray<double>(), new double[10], $"{because}: all_null");
            }
        }

        /// <summary>
        ///     The writer configurations Parquet.Net decodes in full read as numpy's view through <see cref="ParquetFile"/>: the
        ///     dictionary-encoded default and PLAIN values under zstd — the same dtypes, the same bits, across the three row
        ///     groups (rows 0–3, 4–7, 8–9 concatenated in order), the required columns (decoded straight into NDArray memory) and
        ///     the optional ones (read through the null-scatter path) alike. (The DELTA_BINARY_PACKED and data page v2 files are
        ///     pinned by their own tests: Parquet.Net 6.1.0 decodes only part of them.)
        /// </summary>
        /// <param name="file">The fixture written under one writer configuration.</param>
        [TestMethod]
        [DataRow(DefaultFile)]
        [DataRow("numeric_columns_plain_zstd.parquet")]
        public void EveryWriterConfiguration_ReadsAsNumpysView(string file)
        {
            using ParquetFile pf = ParquetFile.Load(Fixture(file));
            pf.RowCount.Should().Be(10);
            pf.RowGroupCount.Should().Be(3);
            pf.Columns.Should().Equal(Loadable, "the loadable columns in schema order: the timestamp and the date have no NumSharp dtype");

            AssertNumericColumnsAreNumpysView(name => pf[name], file);
            pf["flag"].typecode.Should().Be(NPTypeCode.Boolean);
            pf["flag"].ToArray<bool>().Should().Equal(Flag, file);
            AssertNullableColumnsFillDefault(name => pf[name], file);
        }

        /// <summary>
        ///     <c>Stream.ReadParquetAsNDArraysAsync</c> — Parquet.Net's face of the bridge — returns exactly the file loader's
        ///     columns (the same names, in schema order) with the same bytes, in each writer configuration Parquet.Net decodes in
        ///     full.
        /// </summary>
        /// <param name="file">The fixture written under one writer configuration.</param>
        /// <returns>A task that completes when the comparison has run.</returns>
        [TestMethod]
        [DataRow(DefaultFile)]
        [DataRow("numeric_columns_plain_zstd.parquet")]
        public async Task ReadParquetAsNDArraysAsync_IsTheFileLoadersView(string file)
        {
            Dictionary<string, NDArray> columns;
            await using (FileStream stream = File.OpenRead(Fixture(file)))
                columns = await stream.ReadParquetAsNDArraysAsync();

            columns.Keys.Should().Equal(Loadable, file);
            AssertNumericColumnsAreNumpysView(name => columns[name], $"{file} via ReadParquetAsNDArraysAsync");
            using ParquetFile pf = ParquetFile.Load(Fixture(file));
            foreach (string name in Loadable)
            {
                columns[name].typecode.Should().Be(pf[name].typecode, $"{file}: {name}");
                columns[name].tobytes().Should().Equal(pf[name].tobytes(), $"{file}: {name} has the loader's bytes");
            }
        }

        /// <summary>
        ///     The three row groups stream as pyarrow wrote them (4 + 4 + 2 rows), through both faces — the NumSharp face
        ///     (<see cref="ParquetFile.RowGroups"/>) and Parquet.Net's (<c>ParquetRowGroupReader.ReadAsNDArrayAsync</c>) — and for
        ///     EVERY loadable column the groups' bytes, concatenated, are the whole column's bytes: streaming and loading agree
        ///     across the boundaries, nulls and fills included.
        /// </summary>
        /// <returns>A task that completes when both faces have been compared.</returns>
        [TestMethod]
        public async Task RowGroups_EveryColumnStreamsAsTheWholeColumnsSlices_BothFaces()
        {
            using ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile));
            var streamed = Loadable.ToDictionary(name => name, _ => new List<byte>());
            var counts = new List<long>();
            foreach (ParquetRowGroup group in pf.RowGroups)
            {
                using (group)
                {
                    counts.Add(group.RowCount);
                    foreach (string name in Loadable)
                    {
                        NDArray part = group.Column(name);
                        part.size.Should().Be(group.RowCount, name);
                        streamed[name].AddRange(part.tobytes());
                    }
                }
            }

            counts.Should().Equal(new[] { 4L, 4L, 2L }, "row_group_size=4 over ten rows");
            foreach (string name in Loadable)
                streamed[name].Should().Equal(pf[name].tobytes(), $"{name}: the streamed groups concatenate to the whole column");

            await using FileStream stream = File.OpenRead(Fixture(DefaultFile));
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
            DataField[] fields = reader.Schema.GetDataFields();
            foreach (string name in Loadable)
            {
                DataField field = fields.Single(f => f.Name == name);
                var bytes = new List<byte>();
                for (int g = 0; g < reader.RowGroupCount; g++)
                {
                    using ParquetRowGroupReader rg = reader.OpenRowGroupReader(g);
                    bytes.AddRange((await rg.ReadAsNDArrayAsync(field)).tobytes());
                }

                bytes.Should().Equal(pf[name].tobytes(), $"{name}: Parquet.Net's face streams the same bytes");
            }
        }

        /// <summary>
        ///     The bridge's null policy against numpy's view. numpy (through pyarrow) widens a nullable int64 / uint8 to float64
        ///     with NaN, a nullable bool to an object array with None, and reads an all-null float64 as all NaN; the bridge keeps
        ///     each column's dtype and fills <c>default(T)</c> (0, false, 0.0). numpy's view is reachable from the bridge: a NaN
        ///     fill on a float column IS numpy's view (the written <c>-0.0</c> keeps its sign), and an integer column's view is
        ///     rebuilt from a sentinel fill (the bridge returns no null mask).
        /// </summary>
        [TestMethod]
        public void NullableColumns_KeepTheirDtype_AndNumpysNaNViewIsReachable()
        {
            using ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile));
            AssertNullableColumnsFillDefault(name => pf[name], "the default fill");

            ShouldBeBits(pf.Column("maybe_f64", double.NaN).ToArray<double>(), Filled(MaybeF64, double.NaN), "a NaN fill is numpy's maybe_f64");
            ShouldBeBits(pf.Column("all_null", double.NaN).ToArray<double>(), Enumerable.Repeat(double.NaN, 10).ToArray(), "a NaN fill is numpy's all_null");

            // numpy's float64 + NaN view of the nullable int64: fill a sentinel no value uses, widen, and put NaN where it landed.
            NDArray sentinel = pf.Column("maybe_i64", long.MinValue);
            NDArray asNumpy = sentinel.astype(np.float64);
            asNumpy[sentinel == long.MinValue] = double.NaN;
            ShouldBeBits(asNumpy.ToArray<double>(), MaybeI64.Select(v => v.HasValue ? (double)v.Value : double.NaN).ToArray(), "the rebuilt maybe_i64");
        }

        /// <summary>One fill-coercion case: a fill value, the nullable column it fills, and numpy's <c>a[i] = fill</c> answer.</summary>
        /// <param name="Fill">The fill value handed to the bridge.</param>
        /// <param name="Column">The nullable column: <c>maybe_i64</c>, <c>maybe_u8</c>, <c>maybe_flag</c>, <c>maybe_f64</c> or <c>all_null</c>.</param>
        /// <param name="NumpyFill">numpy's value for the fill in the column's dtype; <c>null</c> when numpy refuses the fill.</param>
        /// <param name="Refusal">
        ///     When numpy refuses, the exception the bridge must throw: NumSharp's Python classes, <see cref="OverflowError"/>
        ///     for numpy's <c>OverflowError</c> (a sibling of .NET's <see cref="OverflowException"/> under
        ///     <see cref="ArithmeticException"/>, not one of its subclasses) and <see cref="ValueError"/> for its
        ///     <c>ValueError</c>; <c>null</c> otherwise.
        /// </param>
        /// <param name="Text">numpy's error text for a refusal (LP64 numpy 2.4.2's), or <c>null</c>.</param>
        public readonly record struct FillCase(object Fill, string Column, object NumpyFill, Type Refusal, string Text = null);

        /// <summary>Reads <paramref name="c"/>'s column with its per-call fill and asserts numpy's answer.</summary>
        /// <param name="pf">The open default fixture.</param>
        /// <param name="c">The case.</param>
        private static void AssertFillCase(ParquetFile pf, FillCase c)
        {
            string because = $"fill {(c.Fill is string s ? $"\"{s}\"" : Convert.ToString(c.Fill, CultureInfo.InvariantCulture))} " +
                             $"({c.Fill.GetType().Name}) into {c.Column}";
            if (c.Refusal != null)
            {
                Action read = () => pf.Column(c.Column, c.Fill);
                Exception refusal = read.Should().Throw<Exception>(because).Which;
                refusal.Should().BeAssignableTo(c.Refusal, because);
                if (c.Text != null)
                    refusal.Message.Should().Be(c.Text, because);
                return;
            }

            NDArray nd = pf.Column(c.Column, c.Fill);
            switch (c.Column)
            {
                case "maybe_i64":
                    nd.ToArray<long>().Should().Equal(Filled(MaybeI64, Convert.ToInt64(c.NumpyFill, CultureInfo.InvariantCulture)), because);
                    break;
                case "maybe_u8":
                    nd.ToArray<byte>().Should().Equal(Filled(MaybeU8, Convert.ToByte(c.NumpyFill, CultureInfo.InvariantCulture)), because);
                    break;
                case "maybe_flag":
                    nd.ToArray<bool>().Should().Equal(Filled(MaybeFlag, (bool)c.NumpyFill), because);
                    break;
                case "maybe_f64":
                    ShouldBeBits(nd.ToArray<double>(), Filled(MaybeF64, Convert.ToDouble(c.NumpyFill, CultureInfo.InvariantCulture)), because);
                    break;
                case "all_null":
                    ShouldBeBits(nd.ToArray<double>(), Enumerable.Repeat(Convert.ToDouble(c.NumpyFill, CultureInfo.InvariantCulture), 10).ToArray(), because);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(c), c.Column, "not a nullable column of the fixture");
            }
        }

        /// <summary>
        ///     Every kind of fill converts to the column's dtype as numpy's <c>a[i] = fill</c> converts it: a whole float or text
        ///     into an integer, NaN / a negative / a fraction into a bool (all truthy), text into a float, <c>True</c> into a
        ///     number; NaN or the text <c>'1.5'</c> into an integer refused with numpy's <c>ValueError</c>, an out-of-range value
        ///     into uint8 with numpy's <c>OverflowError</c> (<c>-1</c>, <c>300</c>, and <c>-1.5</c>, which numpy truncates to
        ///     <c>-1</c> before refusing) — the classes and texts NumSharp's <c>PyArray_Pack</c> port raises.
        /// </summary>
        [TestMethod]
        public void NullFill_EveryFillKind_ConvertsAsNumpysSetitem()
        {
            FillCase[] cases =
            {
                new(2.5, "maybe_i64", 2L, null),
                new(-1L, "maybe_i64", -1L, null),
                new("7", "maybe_i64", 7L, null),
                new(true, "maybe_i64", 1L, null),
                new(double.NaN, "maybe_i64", null, typeof(ValueError), "cannot convert float NaN to integer"),
                new("1.5", "maybe_i64", null, typeof(ValueError), "invalid literal for int() with base 10: '1.5'"),
                new("7", "maybe_u8", (byte)7, null),
                new(-1L, "maybe_u8", null, typeof(OverflowError), "Python integer -1 out of bounds for uint8"),
                new(300L, "maybe_u8", null, typeof(OverflowError), "Python integer 300 out of bounds for uint8"),
                new(-1.5, "maybe_u8", null, typeof(OverflowError), "Python integer -1 out of bounds for uint8"),
                new(double.NaN, "maybe_flag", true, null),
                new(-1L, "maybe_flag", true, null),
                new(1.5, "maybe_flag", true, null),
                new(false, "maybe_flag", false, null),
                new("7", "maybe_f64", 7.0, null),
                new("1.5", "maybe_f64", 1.5, null),
                new(true, "maybe_f64", 1.0, null),
                new(double.NaN, "maybe_f64", double.NaN, null),
                new(-0.0, "maybe_f64", -0.0, null),
                new(1e300, "all_null", 1e300, null),
            };
            using ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile));
            foreach (FillCase c in cases)
                AssertFillCase(pf, c);
        }

        /// <summary>
        ///     A fractional fill or a text fill converts by numpy's rules: numpy's <c>a[i] = fill</c> TRUNCATES toward zero
        ///     (<c>1.5 → 1</c>, <c>0.7 → 0</c>, <c>-1.5 → -1</c>) and takes any non-empty text as <c>True</c>.
        ///     <c>ParquetConvert.ResolveFill</c> packs the fill with NumSharp's <c>PyArray_Pack</c> port; it used
        ///     <see cref="Convert.ChangeType(object, Type, IFormatProvider)"/>, which ROUNDED a fraction (<c>1.5 → 2</c>) and
        ///     parsed text as a .NET literal (<c>"7"</c> was no Boolean) — the owner's decision on 2026-11-05: "NumPy's
        ///     a[i] = fill".
        /// </summary>
        [TestMethod]
        public void NullFill_FractionOrText_ConvertsAsNumpysSetitem()
        {
            FillCase[] cases =
            {
                new(1.5, "maybe_i64", 1L, null),
                new(0.7, "maybe_i64", 0L, null),
                new(-1.5, "maybe_i64", -1L, null),
                new(1.5, "maybe_u8", (byte)1, null),
                new("7", "maybe_flag", true, null),
                new("1.5", "maybe_flag", true, null),
            };
            using ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile));
            foreach (FillCase c in cases)
                AssertFillCase(pf, c);
        }

        /// <summary>
        ///     A file-level fill that a column's dtype cannot hold leaves the columns WITHOUT nulls readable: numpy's own NaN
        ///     view of the float columns (<c>NullFill = double.NaN</c>) keeps every integer and decimal column of the file
        ///     readable — the REQUIRED ones, the optional one that holds no null, a zero-row file's — and a text fill
        ///     (<c>"7"</c>) the required boolean. <c>ParquetConvert</c> converts the fill at the first null it meets; it used to
        ///     convert it for every column before reading a value (<see cref="OverflowException"/> /
        ///     <see cref="FormatException"/> on columns with nothing to fill) — the owner's decision on 2026-11-05: "At the
        ///     first null".
        /// </summary>
        [TestMethod]
        public void NullFill_TheDtypeCannotHold_StillReadsColumnsWithoutNulls()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile), new ParquetLoadOptions { NullFill = double.NaN }))
            {
                // What reads today: the float columns take numpy's NaN view.
                ShouldBeBits(pf["maybe_f64"].ToArray<double>(), Filled(MaybeF64, double.NaN), "a file-level NaN fill");
                // What must read: every column without a null is untouched by the fill.
                AssertNumericColumnsAreNumpysView(name => pf[name], "a NaN fill replaces no value of a column without nulls");
            }

            using (ParquetFile empty = ParquetFile.Load(Fixture("empty_columns.parquet"), new ParquetLoadOptions { NullFill = double.NaN }))
                empty["i64"].size.Should().Be(0, "a zero-row column holds no null to fill");

            using (ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile), new ParquetLoadOptions { NullFill = "7" }))
                pf["flag"].ToArray<bool>().Should().Equal(Flag, "a text fill replaces no value of the required boolean");
        }

        /// <summary>
        ///     <see cref="NullHandling.Raise"/> refuses only what actually holds a null: the required columns and the optional
        ///     column without nulls read, each column that has nulls is refused naming it, a row group of a nullable column that
        ///     holds no null streams (the last group of <c>maybe_flag</c>), a per-call fill does not lift the file's Raise, and
        ///     <c>ReadParquetAsNDArraysAsync</c> under Raise refuses the file at its first column with a null.
        /// </summary>
        /// <returns>A task that completes when the extension's refusal has been observed.</returns>
        [TestMethod]
        public async Task Raise_RefusesOnlyWhatHoldsANull()
        {
            var raise = new ParquetLoadOptions { Nulls = NullHandling.Raise };
            using (ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile), raise))
            {
                AssertNumericColumnsAreNumpysView(name => pf[name], "Raise reads the columns without nulls");
                pf["flag"].ToArray<bool>().Should().Equal(Flag);
                foreach (string name in WithNulls)
                {
                    Action read = () => { NDArray _ = pf[name]; };
                    read.Should().Throw<InvalidOperationException>(name)
                        .WithMessage($"Parquet column '{name}' contains null values but NullHandling.Raise was requested.");
                }

                Action perCall = () => pf.Column("maybe_i64", -1L);
                perCall.Should().Throw<InvalidOperationException>("a per-call fill keeps the file's Raise policy");

                ParquetRowGroup[] groups = pf.RowGroups.ToArray();
                try
                {
                    Action first = () => groups[0].Column("maybe_flag");
                    first.Should().Throw<InvalidOperationException>("rows 0-3 of maybe_flag hold a null");
                    groups[2].Column("maybe_flag").ToArray<bool>().Should().Equal(new[] { false, true }, "rows 8-9 hold no null");
                }
                finally
                {
                    foreach (ParquetRowGroup group in groups)
                        group.Dispose();
                }
            }

            await using FileStream stream = File.OpenRead(Fixture(DefaultFile));
            Func<Task> whole = async () => await stream.ReadParquetAsNDArraysAsync(raise);
            (await whole.Should().ThrowAsync<InvalidOperationException>())
                .WithMessage("Parquet column 'maybe_i64' contains null values but NullHandling.Raise was requested.");
        }

        /// <summary>
        ///     A per-call fill (<see cref="ParquetFile.Column(string, object)"/>) reads afresh every time and is never cached,
        ///     while the file-default read is cached (the same instance) and keeps its own fill afterwards;
        ///     <see cref="ParquetLoadOptions.CacheColumns"/> = false reads a new instance per access with the same bytes.
        /// </summary>
        [TestMethod]
        public void PerCallFill_IsFreshAndUncached_TheFileDefaultStaysCached()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile)))
            {
                NDArray cached = pf["maybe_i64"];
                ReferenceEquals(pf["maybe_i64"], cached).Should().BeTrue("the file-default read is cached");
                NDArray once = pf.Column("maybe_i64", -1L);
                NDArray twice = pf.Column("maybe_i64", -1L);
                ReferenceEquals(once, twice).Should().BeFalse("a per-call fill reads afresh");
                once.ToArray<long>().Should().Equal(Filled(MaybeI64, -1L));
                ReferenceEquals(pf["maybe_i64"], cached).Should().BeTrue("the per-call read does not replace the cached entry");
                pf["maybe_i64"].ToArray<long>().Should().Equal(Filled(MaybeI64, 0L));
            }

            using (ParquetFile uncached = ParquetFile.Load(Fixture(DefaultFile), new ParquetLoadOptions { CacheColumns = false }))
            {
                NDArray a = uncached["u64"], b = uncached["u64"];
                ReferenceEquals(a, b).Should().BeFalse("CacheColumns = false reads per access");
                a.tobytes().Should().Equal(b.tobytes());
            }
        }

        /// <summary>
        ///     A projection keeps the loadable columns it names IN SCHEMA ORDER (not the order named) and drops a name that is
        ///     unloadable (the timestamp) or absent from the file without an error; the projected file refuses every other
        ///     column, listing what it holds, and <c>ReadParquetAsNDArraysAsync</c> honours the same projection. (pyarrow's
        ///     <c>read_table(columns=...)</c> keeps the order named and refuses an absent name.)
        /// </summary>
        /// <returns>A task that completes when the extension's projection has been compared.</returns>
        [TestMethod]
        public async Task Projection_KeepsSchemaOrder_AndDropsUnloadableOrAbsentNames()
        {
            var projection = new ParquetLoadOptions { Columns = new[] { "maybe_flag", "when", "nope", "i8" } };
            using (ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile), projection))
            {
                pf.Columns.Should().Equal("i8", "maybe_flag");
                pf["i8"].ToArray<sbyte>().Should().Equal(new sbyte[] { -128, 127, 0, 1, -1, 5, 7, 8, 9, 10 });
                pf["maybe_flag"].ToArray<bool>().Should().Equal(Filled(MaybeFlag, false));
                Action other = () => { NDArray _ = pf["i16"]; };
                other.Should().Throw<ArgumentException>().WithMessage("column 'i16' not found or not loadable. Available: i8, maybe_flag*");
            }

            await using FileStream stream = File.OpenRead(Fixture(DefaultFile));
            Dictionary<string, NDArray> columns = await stream.ReadParquetAsNDArraysAsync(projection);
            columns.Keys.Should().Equal("i8", "maybe_flag");
        }

        /// <summary>
        ///     The timestamp[ms] and date32 columns (Parquet.Net types both <see cref="DateTime"/>; numpy reads datetime64[ms] /
        ///     datetime64[D], dtypes NumSharp does not have) are skipped by the loaders — absent from
        ///     <see cref="ParquetFile.Columns"/>, refused by the indexer as not loadable, <c>TryGetColumn</c> false — and refused
        ///     by every direct reader (by name, by field, per row group) with the "no NumSharp dtype" text, synchronously, when
        ///     the reader is CALLED (before a task exists). An absent name is refused the same way.
        /// </summary>
        /// <returns>A task that completes when every reader has been tried.</returns>
        [TestMethod]
        public async Task TemporalColumns_AreSkippedByTheLoaders_AndRefusedByTheDirectReaders()
        {
            string[] temporal = { "when", "day" };
            using (ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile)))
            {
                foreach (string name in temporal)
                {
                    pf.TryGetColumn(name, out NDArray none).Should().BeFalse(name);
                    none.Should().BeNull();
                    Action index = () => { NDArray _ = pf[name]; };
                    index.Should().Throw<ArgumentException>(name).WithMessage($"column '{name}' not found or not loadable. Available: i8, i16, *");
                }
            }

            await using FileStream stream = File.OpenRead(Fixture(DefaultFile));
            await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
            foreach (string name in temporal)
            {
                DataField field = reader.Schema.GetDataFields().Single(f => f.Name == name);
                field.ClrType.Should().Be(typeof(DateTime), name);
                string refusal = string.Format(CultureInfo.InvariantCulture, NoDtype, name);

                Action byName = () => reader.ReadColumnAsNDArrayAsync(name);
                byName.Should().Throw<NotSupportedException>(name).WithMessage(refusal);
                Action byField = () => reader.ReadColumnAsNDArrayAsync(field);
                byField.Should().Throw<NotSupportedException>(name).WithMessage(refusal);
                using ParquetRowGroupReader rg = reader.OpenRowGroupReader(0);
                Action perGroup = () => rg.ReadAsNDArrayAsync(field);
                perGroup.Should().Throw<NotSupportedException>(name).WithMessage(refusal);
            }

            Action absent = () => reader.ReadColumnAsNDArrayAsync("nope");
            absent.Should().Throw<ArgumentException>().WithMessage("column 'nope' not found*");
        }

        /// <summary>
        ///     A zero-row file (one empty row group) reads every loadable column as an EMPTY 1-D array of its own dtype — numpy's
        ///     view of an empty column, the nullable ones included (no null, so pyarrow keeps int64 / uint8 / bool) — through the
        ///     loader, the extension and the stream of row groups, under the default and the Raise policies.
        /// </summary>
        /// <returns>A task that completes when the extension's columns have been checked.</returns>
        [TestMethod]
        public async Task EmptyFile_EveryColumnIsAnEmptyArrayOfItsDtype()
        {
            var dtypes = new Dictionary<string, NPTypeCode>
            {
                ["i8"] = NPTypeCode.SByte, ["i16"] = NPTypeCode.Int16, ["i32"] = NPTypeCode.Int32, ["i64"] = NPTypeCode.Int64,
                ["u8"] = NPTypeCode.Byte, ["u16"] = NPTypeCode.UInt16, ["u32"] = NPTypeCode.UInt32, ["u64"] = NPTypeCode.UInt64,
                ["f32"] = NPTypeCode.Single, ["f64"] = NPTypeCode.Double, ["flag"] = NPTypeCode.Boolean, ["dec"] = NPTypeCode.Decimal,
                ["opt_i32"] = NPTypeCode.Int32, ["maybe_i64"] = NPTypeCode.Int64, ["maybe_f64"] = NPTypeCode.Double,
                ["maybe_u8"] = NPTypeCode.Byte, ["maybe_flag"] = NPTypeCode.Boolean, ["all_null"] = NPTypeCode.Double,
            };
            dtypes.Keys.Should().Equal(Loadable);

            foreach (ParquetLoadOptions options in new[] { null, new ParquetLoadOptions { Nulls = NullHandling.Raise } })
            {
                using ParquetFile pf = ParquetFile.Load(Fixture("empty_columns.parquet"), options);
                pf.RowCount.Should().Be(0);
                pf.RowGroupCount.Should().Be(1, "pyarrow writes one empty row group");
                pf.Columns.Should().Equal(Loadable);
                foreach (string name in Loadable)
                {
                    NDArray nd = pf[name];
                    nd.typecode.Should().Be(dtypes[name], name);
                    nd.shape.Should().Equal(new long[] { 0 }, name);
                }

                foreach (ParquetRowGroup group in pf.RowGroups)
                {
                    using (group)
                    {
                        group.RowCount.Should().Be(0);
                        foreach (string name in Loadable)
                            group.Column(name).shape.Should().Equal(new long[] { 0 }, name);
                    }
                }
            }

            await using FileStream stream = File.OpenRead(Fixture("empty_columns.parquet"));
            Dictionary<string, NDArray> columns = await stream.ReadParquetAsNDArraysAsync();
            columns.Keys.Should().Equal(Loadable);
            columns.Values.Should().AllSatisfy(nd => nd.size.Should().Be(0));
        }

        /// <summary>
        ///     The decimals .NET's 96-bit decimal holds read exactly — the extremes of a decimal128(29, 0)
        ///     (±79228162514264337593543950335, <see cref="decimal.MaxValue"/> / <see cref="decimal.MinValue"/>) and a
        ///     decimal128(10, 10) with its ten-digit scale (numpy prints the last as <c>Decimal('-1E-10')</c>, the same value and
        ///     exponent).
        /// </summary>
        [TestMethod]
        public void WideDecimals_WithinDotNetDecimal_AreExact()
        {
            using ParquetFile pf = ParquetFile.Load(Fixture("wide_decimals.parquet"));
            pf.Columns.Should().Equal("d29_0_max", "d10_10", "d38_4_precise", "d29_0_over", "d38_0_huge");
            pf["d29_0_max"].ToArray<decimal>().Should().Equal(decimal.MaxValue, decimal.MinValue);
            decimal[] tiny = pf["d10_10"].ToArray<decimal>();
            tiny.Should().Equal(0.1234567891m, -0.0000000001m);
            tiny.Select(d => d.Scale).Should().Equal(new byte[] { 10, 10 }, "decimal128(10, 10)'s scale");
        }

        /// <summary>
        ///     MISALIGNED (the decimal dtype's range): numpy reads a Parquet decimal as exact Python <c>Decimal</c>s of any
        ///     precision (an object array), while NumSharp's decimal dtype is .NET's 96-bit decimal (28-29 significant digits, at
        ///     most ±79228162514264337593543950335). Parquet.Net ROUNDS a decimal128(38, 4) value past 29 digits
        ///     (<c>1234567890123456789012345678.1234</c> reads <c>...678.1</c>) and refuses one past the range with
        ///     <see cref="OverflowException"/> — and that one column fails <c>ReadParquetAsNDArraysAsync</c> for the whole file.
        /// </summary>
        /// <returns>A task that completes when the extension's refusal has been observed.</returns>
        [TestMethod]
        [TestCategory("Misaligned")]
        public async Task WideDecimals_PastDotNetDecimal_RoundOrOverflow_WhereNumpyKeepsExactDecimals()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture("wide_decimals.parquet")))
            {
                pf["d38_4_precise"].ToArray<decimal>().Select(d => d.ToString(CultureInfo.InvariantCulture))
                    .Should().Equal(new[] { "1234567890123456789012345678.1", "1.0001" }, "numpy: Decimal('1234567890123456789012345678.1234')");
                foreach (string name in new[] { "d29_0_over", "d38_0_huge" })
                {
                    Action read = () => { NDArray _ = pf[name]; };
                    read.Should().Throw<OverflowException>(name);
                }
            }

            await using FileStream stream = File.OpenRead(Fixture("wide_decimals.parquet"));
            Func<Task> whole = async () => await stream.ReadParquetAsNDArraysAsync();
            await whole.Should().ThrowAsync<OverflowException>("one decimal past the range fails the whole read");
        }

        /// <summary>
        ///     MISALIGNED (Parquet.Net 6.1.0): pyarrow's data page v2 writes booleans RLE-encoded, and Parquet.Net 6.1.0 cannot
        ///     decode an RLE boolean page — its own serializer fails on the same file — so reading the required <c>flag</c> or the
        ///     nullable <c>maybe_flag</c> surfaces Parquet.Net's <see cref="DivideByZeroException"/>, and the whole-file
        ///     extension fails with it, where numpy reads both. Every other column of the v2 file reads numpy's view.
        /// </summary>
        /// <returns>A task that completes when the extension's failure has been observed.</returns>
        [TestMethod]
        [TestCategory("Misaligned")]
        public async Task DataPageV2Booleans_ParquetNetCannotDecode_EveryOtherColumnReadsNumpysView()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture(PageV2File)))
            {
                pf.Columns.Should().Equal(Loadable, "the booleans are loadable by type; the failure is in the decode");
                foreach (string name in new[] { "flag", "maybe_flag" })
                {
                    Action read = () => { NDArray _ = pf[name]; };
                    read.Should().Throw<DivideByZeroException>(name);
                }

                AssertNumericColumnsAreNumpysView(name => pf[name], PageV2File);
                ShouldBeBits(pf["maybe_f64"].ToArray<double>(), Filled(MaybeF64, 0.0), PageV2File);
                pf["maybe_i64"].ToArray<long>().Should().Equal(Filled(MaybeI64, 0L));
            }

            await using FileStream stream = File.OpenRead(Fixture(PageV2File));
            Func<Task> whole = async () => await stream.ReadParquetAsNDArraysAsync();
            await whole.Should().ThrowAsync<DivideByZeroException>("the whole-file read stops at flag");
        }

        /// <summary>
        ///     MISALIGNED (Parquet.Net 6.1.0): DELTA_BINARY_PACKED — the integer encoding Parquet's v2 writers (Spark /
        ///     parquet-mr) choose by default — is decoded by Parquet.Net into int16, uint16, int32, uint32, int64 and uint64, but
        ///     an int8 or uint8 column (physically INT32) and a decimal stored as an INT64 are refused with
        ///     <see cref="NotSupportedException"/> (<c>element type System.SByte is not supported in DELTA_BINARY_PACKED</c>),
        ///     and the whole-file extension fails at the first of them, where numpy reads every column. Every other column of
        ///     the file — the delta int16..uint64 and the BYTE_STREAM_SPLIT floats included — reads numpy's view.
        /// </summary>
        /// <returns>A task that completes when the extension's refusal has been observed.</returns>
        [TestMethod]
        [TestCategory("Misaligned")]
        public async Task DeltaBinaryPacked_Int8Uint8Decimal_ParquetNetCannotDecode_EveryOtherColumnReadsNumpysView()
        {
            using (ParquetFile pf = ParquetFile.Load(Fixture(DeltaFile)))
            {
                pf.Columns.Should().Equal(Loadable, "the refused columns are loadable by type; the failure is in the decode");
                foreach ((string name, string clr) in new[] { ("i8", "SByte"), ("u8", "Byte"), ("maybe_u8", "Byte"), ("dec", "Decimal") })
                {
                    Action read = () => { NDArray _ = pf[name]; };
                    read.Should().Throw<NotSupportedException>(name).WithMessage($"element type System.{clr} is not supported in DELTA_BINARY_PACKED");
                }

                AssertNumericColumnsAreNumpysView(name => pf[name], DeltaFile, "i8", "u8", "dec");
                pf["flag"].ToArray<bool>().Should().Equal(Flag, DeltaFile);
                AssertNullableColumnsFillDefault(name => pf[name], DeltaFile, "maybe_u8");
            }

            await using FileStream stream = File.OpenRead(Fixture(DeltaFile));
            Func<Task> whole = async () => await stream.ReadParquetAsNDArraysAsync();
            (await whole.Should().ThrowAsync<NotSupportedException>("the whole-file read stops at i8"))
                .WithMessage("element type System.SByte is not supported in DELTA_BINARY_PACKED");
        }

        /// <summary>
        ///     MISALIGNED (Parquet.Net 6.1.0): Parquet.Net's decimal decode drops the trailing zeros of the column's scale — a
        ///     decimal128(10, 2) <c>-3.50</c> reads <c>-3.5</c>, <c>0.00</c> reads <c>0</c> — where numpy keeps the exponent
        ///     (<c>Decimal('-3.50')</c>). The values are equal; only the scale, and with it the printed text, differ. NumSharp's
        ///     own decimal storage keeps a scale (<c>np.array(new[] { -3.50m })</c> prints <c>-3.50</c>), so the loss is the
        ///     reader's.
        /// </summary>
        [TestMethod]
        [TestCategory("Misaligned")]
        public void Decimal_TrailingZerosOfTheScale_DropInParquetNetsDecode()
        {
            using ParquetFile pf = ParquetFile.Load(Fixture(DefaultFile));
            pf["dec"].ToArray<decimal>().Select(d => d.ToString(CultureInfo.InvariantCulture))
                .Should().Equal(new[] { "1.25", "-3.5", "0", "99999999.99", "-0.01", "7", "8.1", "9.99", "10", "-12345678.9" },
                    $"numpy prints {string.Join(", ", Decimals)}");
            np.array(new[] { -3.50m, 0.00m }).ToArray<decimal>().Select(d => d.ToString(CultureInfo.InvariantCulture))
                .Should().Equal(new[] { "-3.50", "0.00" }, "NumSharp's decimal storage keeps the scale it is given");
        }

        /// <summary>
        ///     MISALIGNED (Parquet.Net 6.1.0): a float16 column (pyarrow's FLOAT16 logical type) makes Parquet.Net refuse the
        ///     file's SCHEMA (<c>type System.Half is not supported</c>), so neither the float16 column nor the int32 beside it
        ///     can be read — numpy reads both, and NumSharp has the float16 dtype. The failed <see cref="ParquetFile.Load(string, ParquetLoadOptions)"/>
        ///     releases the file it opened (an exclusive open succeeds afterwards).
        /// </summary>
        /// <returns>A task that completes when the extension's refusal has been observed.</returns>
        [TestMethod]
        [TestCategory("Misaligned")]
        public async Task Float16Column_ParquetNetRefusesTheWholeSchema()
        {
            string path = Fixture("float16_column.parquet");
            Action load = () => ParquetFile.Load(path);
            load.Should().Throw<NotSupportedException>().WithMessage("type System.Half is not supported");
            using (new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None))
            {
                // Opening with FileShare.None succeeds only when no handle to the file is left open.
            }

            await using FileStream stream = File.OpenRead(path);
            Func<Task> whole = async () => await stream.ReadParquetAsNDArraysAsync();
            (await whole.Should().ThrowAsync<NotSupportedException>()).WithMessage("type System.Half is not supported");
        }

        /// <summary>
        ///     A file that is not Parquet is refused by <see cref="ParquetFile.Load(string, ParquetLoadOptions)"/> with
        ///     Parquet.Net's own error, and the failed load releases the file it opened (it can be deleted at once).
        /// </summary>
        [TestMethod]
        public void Load_NotParquet_IsRefused_AndReleasesTheFile()
        {
            string path = Path.Combine(Path.GetTempPath(), $"nsparquet_notparquet_{Guid.NewGuid():N}.parquet");
            File.WriteAllBytes(path, new byte[] { 0x50, 0x41, 0x52, 0x31, 1, 2, 3, 4, 5, 6, 7, 8 });   // "PAR1" then no footer
            try
            {
                Action load = () => ParquetFile.Load(path);
                load.Should().Throw<IOException>();
            }
            finally
            {
                // Deleting succeeds only when the failed load closed its stream.
                File.Delete(path);
            }

            File.Exists(path).Should().BeFalse();
        }
    }
}
