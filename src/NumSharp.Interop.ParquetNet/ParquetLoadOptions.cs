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
        /// <c>default(T)</c> of the column's dtype (0 / false; the NA of a text column's <c>StringDType(na_object=None)</c>,
        /// numpy's <c>None</c>; <c>b""</c> in a binary column's <c>S</c> array, which has no NA). This is the default policy.
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
        /// A non-null value converts into each column it fills as NumPy's <c>a[i] = fill</c> does (NumSharp's
        /// <c>PyArray_Pack</c>): a float truncates toward zero into an integer column, NaN or an infinity into one is
        /// refused (<c>ValueError</c> / <c>OverflowException</c>), an integer out of the dtype's range is refused with
        /// NumPy's <c>OverflowError</c> text, a string is parsed (any non-empty string is <c>True</c> in a boolean
        /// column), and into a text column a number becomes its text (NaN is <c>'nan'</c>).
        /// </summary>
        /// <remarks>
        /// The fill converts only where a null is: a column without nulls reads untouched under any fill, so NumPy's
        /// NaN view of the float columns (<c>NullFill = double.NaN</c>) leaves the integer columns of the same file
        /// readable; an integer column WITH a null then refuses the NaN, as NumPy's <c>a[mask] = nan</c> does.
        /// </remarks>
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
