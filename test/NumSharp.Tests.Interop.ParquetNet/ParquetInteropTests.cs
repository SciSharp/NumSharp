using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.ParquetNet;
using Parquet;                       // extension methods (ReadParquetAsNDArraysAsync, AsMemory, ...)
using Parquet.Serialization;         // ParquetSerializer to WRITE the sample files

namespace NumSharp.Tests.Interop.ParquetNet
{
    /// <summary>
    /// End-to-end tests: write a sample Parquet file with Parquet.Net's serializer, then read its columns
    /// back into NDArrays through the interop package and assert values, dtypes, null-fill, multi-row-group
    /// concatenation, streaming, projection and the extension surface.
    /// </summary>
    [TestClass]
    public class ParquetInteropTests
    {
        /// <summary>A POCO covering several supported dtypes plus a nullable column.</summary>
        public class RowPoco
        {
            public int Id { get; set; }
            public double Price { get; set; }
            public long Qty { get; set; }
            public bool Active { get; set; }
            public float Ratio { get; set; }
            public int? Maybe { get; set; }   // nullable -> exercises the definition-level / fill path
        }

        // Write `count` rows to a temp .parquet file; optionally force multiple row groups.
        private static string WriteSample(int count, int rowGroupSize = 0)
        {
            var rows = Enumerable.Range(0, count).Select(i => new RowPoco
            {
                Id = i,
                Price = i * 1.5,
                Qty = i * 100L,
                Active = (i % 2 == 0),
                Ratio = i * 0.25f,
                Maybe = (i % 3 == 0) ? (int?)null : i
            }).ToList();

            string path = Path.Combine(Path.GetTempPath(), $"nsparquet_{Guid.NewGuid():N}.parquet");
            // Parquet.Net's serializer takes Parquet.ParquetOptions (its RowGroupSize forces multiple row
            // groups). This is the very type whose name we avoided colliding with: ours is ParquetLoadOptions.
            ParquetOptions opts = rowGroupSize > 0 ? new ParquetOptions { RowGroupSize = rowGroupSize } : null;
            ParquetSerializer.SerializeAsync(rows, path, opts).GetAwaiter().GetResult();
            return path;
        }

        [TestMethod]
        public void Load_AllSupportedDtypes_MatchesValues()
        {
            string path = WriteSample(100);
            try
            {
                using var pf = ParquetFile.Load(path);
                pf.RowCount.Should().Be(100);

                NDArray id = pf["Id"];
                id.typecode.Should().Be(NPTypeCode.Int32);
                id.size.Should().Be(100);
                id.ToArray<int>().Should().Equal(Enumerable.Range(0, 100).ToArray());

                pf["Price"].typecode.Should().Be(NPTypeCode.Double);
                pf["Price"].ToArray<double>().Should().Equal(Enumerable.Range(0, 100).Select(i => i * 1.5).ToArray());

                pf["Qty"].ToArray<long>().Should().Equal(Enumerable.Range(0, 100).Select(i => i * 100L).ToArray());
                pf["Active"].ToArray<bool>().Should().Equal(Enumerable.Range(0, 100).Select(i => i % 2 == 0).ToArray());
                pf["Ratio"].ToArray<float>().Should().Equal(Enumerable.Range(0, 100).Select(i => i * 0.25f).ToArray());
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Load_NullableColumn_FillsWithDefault()
        {
            string path = WriteSample(100);
            try
            {
                using var pf = ParquetFile.Load(path);
                NDArray maybe = pf["Maybe"];
                maybe.typecode.Should().Be(NPTypeCode.Int32);
                int[] expected = Enumerable.Range(0, 100).Select(i => (i % 3 == 0) ? 0 : i).ToArray();
                maybe.ToArray<int>().Should().Equal(expected);
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Load_NullableColumn_CustomFill()
        {
            string path = WriteSample(30);
            try
            {
                using var pf = ParquetFile.Load(path);
                NDArray maybe = pf.Column("Maybe", nullFill: -1);
                int[] expected = Enumerable.Range(0, 30).Select(i => (i % 3 == 0) ? -1 : i).ToArray();
                maybe.ToArray<int>().Should().Equal(expected);
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Load_NullableColumn_RaisePolicy_Throws()
        {
            string path = WriteSample(30);
            try
            {
                using var pf = ParquetFile.Load(path, new ParquetLoadOptions { Nulls = NullHandling.Raise });
                Action act = () => { NDArray _ = pf["Maybe"]; };
                act.Should().Throw<InvalidOperationException>();
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Load_MultipleRowGroups_ConcatenatesCorrectly()
        {
            string path = WriteSample(100, rowGroupSize: 40);   // -> 3 row groups (40, 40, 20)
            try
            {
                using var pf = ParquetFile.Load(path);
                pf.RowGroupCount.Should().BeGreaterThan(1);
                pf["Id"].ToArray<int>().Should().Equal(Enumerable.Range(0, 100).ToArray());
                pf["Price"].size.Should().Be(100);
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void RowGroups_Streaming_YieldsAllRows()
        {
            string path = WriteSample(100, rowGroupSize: 40);
            try
            {
                using var pf = ParquetFile.Load(path);
                long total = 0;
                var seen = new List<int>();
                foreach (ParquetRowGroup g in pf.RowGroups)
                {
                    using (g)
                    {
                        total += g.RowCount;
                        seen.AddRange(g.Column("Id").ToArray<int>());
                    }
                }
                total.Should().Be(100);
                seen.Should().Equal(Enumerable.Range(0, 100).ToArray());
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Extension_ReadParquetAsNDArraysAsync_Works()
        {
            string path = WriteSample(50);
            try
            {
                using var fs = File.OpenRead(path);
                Dictionary<string, NDArray> cols = fs.ReadParquetAsNDArraysAsync().GetAwaiter().GetResult();
                cols.Should().ContainKey("Id");
                cols.Should().ContainKey("Price");
                cols["Id"].ToArray<int>().Should().Equal(Enumerable.Range(0, 50).ToArray());
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public async Task Extension_ReadColumnAsNDArray_OnReader_Works()
        {
            string path = WriteSample(20);
            try
            {
                using var fs = File.OpenRead(path);
                await using ParquetReader reader = await ParquetReader.CreateAsync(fs);   // IAsyncDisposable
                NDArray qty = await reader.ReadColumnAsNDArrayAsync("Qty");
                qty.ToArray<long>().Should().Equal(Enumerable.Range(0, 20).Select(i => i * 100L).ToArray());
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Projection_LimitsColumns_AndRejectsOthers()
        {
            string path = WriteSample(10);
            try
            {
                using var pf = ParquetFile.Load(path, new ParquetLoadOptions { Columns = new[] { "Id", "Price" } });
                pf.Columns.Should().BeEquivalentTo(new[] { "Id", "Price" });

                Action act = () => { NDArray _ = pf["Qty"]; };
                act.Should().Throw<ArgumentException>();
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void BagObj_DotAccess_Works()
        {
            string path = WriteSample(10);
            try
            {
                using var pf = ParquetFile.Load(path);
                NDArray id = pf.f.Id;
                id.ToArray<int>().Should().Equal(Enumerable.Range(0, 10).ToArray());
            }
            finally { File.Delete(path); }
        }

        [TestMethod]
        public void Column_Caching_ReturnsSameInstance()
        {
            string path = WriteSample(10);
            try
            {
                using var pf = ParquetFile.Load(path);
                NDArray a = pf["Id"];
                NDArray b = pf["Id"];
                ReferenceEquals(a, b).Should().BeTrue();
            }
            finally { File.Delete(path); }
        }
    }
}
