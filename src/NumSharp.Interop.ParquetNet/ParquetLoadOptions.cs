namespace NumSharp.Interop.ParquetNet
{
    /// <summary>
    /// How null values in a nullable Parquet column are represented once loaded into an NDArray, which has no
    /// native "NA". The default is <see cref="Fill"/>.
    /// </summary>
    public enum NullHandling
    {
        /// <summary>
        /// Replace nulls with a fill value: <see cref="ParquetLoadOptions.NullFill"/> when provided, otherwise
        /// <c>default(T)</c> of the column's dtype (0 / false). This is the default policy.
        /// </summary>
        Fill,

        /// <summary>Throw when a column actually contains one or more nulls.</summary>
        Raise
    }

    /// <summary>
    /// Options controlling how a Parquet file / column is loaded into NDArrays.
    ///
    /// <para>Named <c>ParquetLoadOptions</c> (not <c>ParquetOptions</c>) on purpose: Parquet.Net already ships a
    /// public <c>Parquet.ParquetOptions</c>, and a same-named type would be ambiguous in files that
    /// <c>using Parquet;</c>.</para>
    /// </summary>
    public sealed class ParquetLoadOptions
    {
        /// <summary>
        /// Column projection: when set, only these columns are exposed and loaded (Parquet only touches the
        /// requested column chunks). <c>null</c> = all loadable columns.
        /// </summary>
        public string[] Columns { get; set; }

        /// <summary>Null policy for nullable columns. Default <see cref="NullHandling.Fill"/>.</summary>
        public NullHandling Nulls { get; set; } = NullHandling.Fill;

        /// <summary>
        /// Fill value for <see cref="NullHandling.Fill"/>. <c>null</c> ⇒ <c>default(T)</c> of the column dtype.
        /// A non-null value is coerced to the column's element type (invariant culture).
        /// </summary>
        public object NullFill { get; set; }

        /// <summary>
        /// When true (default), a column materialized through <see cref="ParquetFile"/> is cached so repeated
        /// access returns the same NDArray. Set false for a read-once, low-retention mode.
        /// </summary>
        public bool CacheColumns { get; set; } = true;

        /// <summary>Returns <paramref name="options"/> or a fresh default instance when it is null.</summary>
        internal static ParquetLoadOptions OrDefault(ParquetLoadOptions options) => options ?? new ParquetLoadOptions();
    }
}
