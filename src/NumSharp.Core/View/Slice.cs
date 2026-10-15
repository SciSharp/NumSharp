using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

namespace NumSharp
{
    /// <summary>                                                                                                                                         <br></br>
    /// NDArray can be indexed using slicing                                                                                                              <br></br>
    /// A slice is constructed by start:stop:step notation                                                                                                <br></br>
    ///                                                                                                                                                   <br></br>
    /// Examples:                                                                                                                                         <br></br>
    ///                                                                                                                                                   <br></br>
    /// a[start:stop]  # items start through stop-1                                                                                                       <br></br>
    /// a[start:]      # items start through the rest of the array                                                                                        <br></br>
    /// a[:stop]       # items from the beginning through stop-1                                                                                          <br></br>
    ///                                                                                                                                                   <br></br>
    /// The key point to remember is that the :stop value represents the first value that is not                                                          <br></br>
    /// in the selected slice. So, the difference between stop and start is the number of elements                                                        <br></br>
    /// selected (if step is 1, the default).                                                                                                             <br></br>
    ///                                                                                                                                                   <br></br>
    /// There is also the step value, which can be used with any of the above:                                                                            <br></br>
    /// a[:]           # a copy of the whole array                                                                                                        <br></br>
    /// a[start:stop:step] # start through not past stop, by step                                                                                         <br></br>
    ///                                                                                                                                                   <br></br>
    /// The other feature is that start or stop may be a negative number, which means it counts                                                           <br></br>
    /// from the end of the array instead of the beginning. So:                                                                                           <br></br>
    /// a[-1]    # last item in the array                                                                                                                 <br></br>
    /// a[-2:]   # last two items in the array                                                                                                            <br></br>
    /// a[:-2]   # everything except the last two items                                                                                                   <br></br>
    /// Similarly, step may be a negative number:                                                                                                         <br></br>
    ///                                                                                                                                                   <br></br>
    /// a[::- 1]    # all items in the array, reversed                                                                                                    <br></br>
    /// a[1::- 1]   # the first two items, reversed                                                                                                       <br></br>
    /// a[:-3:-1]  # the last two items, reversed                                                                                                         <br></br>
    /// a[-3::- 1]  # everything except the last two items, reversed                                                                                      <br></br>
    ///                                                                                                                                                   <br></br>
    /// NumSharp is kind to the programmer if there are fewer items than                                                                                  <br></br>
    /// you ask for. For example, if you  ask for a[:-2] and a only contains one element, you get an                                                      <br></br>
    /// empty list instead of an error.Sometimes you would prefer the error, so you have to be aware                                                      <br></br>
    /// that this may happen.                                                                                                                             <br></br>
    ///                                                                                                                                                   <br></br>
    /// Adapted from Greg Hewgill's answer on Stackoverflow: https://stackoverflow.com/questions/509211/understanding-slice-notation                      <br></br>
    ///                                                                                                                                                   <br></br>
    /// Note: special IsIndex == true                                                                                                                     <br></br>
    /// It will pick only a single value at Start in this dimension effectively reducing the Shape of the sliced matrix by 1 dimension.                   <br></br>
    /// It can be used to reduce an N-dimensional array/matrix to a (N-1)-dimensional array/matrix                                                        <br></br>
    ///                                                                                                                                                   <br></br>
    /// Example:                                                                                                                                          <br></br>
    /// a=[[1, 2], [3, 4]]                                                                                                                                <br></br>
    /// a[:, 1] returns the second column of that 2x2 matrix as a 1-D vector                                                                              <br></br>
    /// </summary>
    [DebuggerStepThrough]
    public partial class Slice : IIndex
    {
        /// <summary>
        /// return : for this dimension
        /// </summary>
        public static readonly Slice All = new Slice(null, null);

        /// <summary>
        /// return 0:0 for this dimension
        /// </summary>
        public static readonly Slice None = new Slice(0, 0, 1);

        /// <summary>
        /// fill up the missing dimensions with : at this point, corresponds to ... 
        /// </summary>
        public static readonly Slice Ellipsis = new Slice(0, 0, 1) { IsEllipsis = true };

        /// <summary>
        /// insert a new dimension at this point
        /// </summary>
        public static readonly Slice NewAxis = new Slice(0, 0, 1) { IsNewAxis = true };

        /// <summary>
        /// return exactly one element at this dimension and reduce the shape from n-dim to (n-1)-dim
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [MethodImpl(Inline)]
        public static Slice Index(long index) => new Slice(index, index + 1) { IsIndex = true };

        /// <summary>
        /// Backwards-compatible overload accepting int index.
        /// </summary>
        [MethodImpl(Inline)]
        public static Slice Index(int index) => Index((long)index);

        ///// <summary>
        ///// return multiple elements for this dimension specified by the given index array (or boolean mask array)
        ///// </summary>
        ///// <param name="index_array_or_mask"></param>
        ///// <returns></returns>
        //[MethodImpl(Inline)]
        //public static Slice Select(NDArray index_array_or_mask) => new Slice(null, null) { Selection=index_array_or_mask };

        public long? Start;
        public long? Stop;
        public long Step;
        public bool IsIndex;
        public bool IsEllipsis;
        public bool IsNewAxis;

        ///// <summary>
        ///// Array of integer indices to select elements by index extraction or boolean values to select by masking the elements of the given dimension.
        ///// </summary>
        //public NDArray Selection = null;

        /// <summary>
        /// Length of the slice.
        /// <remarks>
        /// The length is not guaranteed to be known for i.e. a slice like ":". Make sure to check Start and Stop
        /// for null before using it</remarks>
        /// </summary>
        public long? Length => Stop - Start;

        /// <summary>
        /// ndarray can be indexed using slicing
        /// slice is constructed by start:stop:step notation
        /// </summary>
        /// <param name="start">Start index of the slice, null means from the start of the array</param>
        /// <param name="stop">Stop index (first index after end of slice), null means to the end of the array</param>
        /// <param name="step">Optional step to select every n-th element, defaults to 1</param>
        public Slice(long? start = null, long? stop = null, long step = 1)
        {
            Start = start;
            Stop = stop;
            Step = step;
        }

        /// <summary>
        /// Backwards-compatible constructor accepting int parameters.
        /// </summary>
        public Slice(int? start, int? stop, int step = 1) : this((long?)start, (long?)stop, (long)step)
        {
        }

        public Slice(string slice_notation)
        {
            Parse(slice_notation);
        }

        /// <summary>
        /// Parses Python array slice notation and returns an array of Slice objects
        /// </summary>
        public static Slice[] ParseSlices(string multi_slice_notation)
        {
            return DimensionSeparatorRegex().Split(multi_slice_notation).Where(s => !string.IsNullOrWhiteSpace(s)).Select(token => new Slice(token)).ToArray();
        }

        // ---------------------------------------------------------------------------------------------------------
        // Slice-notation grammar. Every pattern is a [GeneratedRegex] singleton, never a static
        // Regex.Split/Match/Replace(input, pattern) call — see SliceNotationRegex's remarks for why.
        // ---------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The separator between the per-dimension terms of a multi-dimensional slice string: a comma plus any
        ///     whitespace after it (<c>,\s*</c>), as <see cref="ParseSlices"/> splits <c>"1:3, ::2"</c>.
        /// </summary>
        /// <returns>The process-wide source-generated <see cref="Regex"/> singleton for <c>,\s*</c>.</returns>
        /// <remarks>Generated rather than cached for the reason given on <see cref="SliceNotationRegex"/>.</remarks>
        [GeneratedRegex(@",\s*")]
        private static partial Regex DimensionSeparatorRegex();

        /// <summary>
        ///     One dimension's slice term: <c>start:stop:step</c> with every part optional (each part a signed integer
        ///     that may carry spaces after its sign, as Python allows <c>"- 1"</c>), a bare index, <c>...</c>, or
        ///     <c>newaxis</c>/<c>np.newaxis</c> — captured into the <c>start</c>/<c>stop</c>/<c>step</c>/<c>index</c>/
        ///     <c>ellipsis</c>/<c>newaxis</c> groups <see cref="Parse"/> reads.
        /// </summary>
        /// <returns>The process-wide source-generated <see cref="Regex"/> singleton for the notation pattern.</returns>
        /// <remarks>
        ///     <para>
        ///     Why generated singletons and not the static <see cref="Regex"/> helpers: the static helpers resolve the
        ///     pattern through the runtime's process-wide pattern cache (<see cref="Regex.CacheSize"/>, 15 entries),
        ///     and on .NET 8 that cache THRASHES once it is full of stale entries. Its recency is a Lamport-style
        ///     stamp — a hit sets the entry's stamp to the last-accessed entry's stamp + 1 — and a newly added entry
        ///     starts at stamp 0, so when every resident entry carries a high stamp each miss evicts the PREVIOUS
        ///     newcomer. Slice parsing cycles through three patterns, so from then on it misses on every call and
        ///     re-parses this (large) pattern each time: measured on .NET 8.0.29, <see cref="ParseSlices"/> went from
        ///     0.6–2.8 µs to 9–18 µs per call, and a 1,000-step training loop that slices with strings ran 4× slower
        ///     (the Karpathy MicroGPT demo: 0.6 s → 2.5 s inside a full test run). Reaching that state needs only a
        ///     long-lived process whose cache filled up and a key change — the cache key includes
        ///     <c>CultureInfo.CurrentCulture</c>, so switching culture turns all three patterns into new keys.
        ///     .NET 10's cache does not thrash, but the singletons are faster there too: no culture lookup, key build or
        ///     dictionary probe per call, and generated matching code instead of the interpreter.
        ///     </para>
        ///     <para>
        ///     Semantics are identical to the static helpers: <see cref="RegexOptions.None"/> (no IgnoreCase, so the
        ///     construction culture never influenced matching) and the same default match timeout
        ///     (<c>REGEX_DEFAULT_MATCH_TIMEOUT</c>, infinite unless the host sets it).
        ///     </para>
        /// </remarks>
        [GeneratedRegex(@"^\s*((?'start'[+-]?\s*\d+)?\s*:\s*(?'stop'[+-]?\s*\d+)?\s*(:\s*(?'step'[+-]?\s*\d+)?)?|(?'index'[+-]?\s*\d+)|(?'ellipsis'\.\.\.)|(?'newaxis'(np\.)?newaxis))\s*$")]
        private static partial Regex SliceNotationRegex();

        /// <summary>
        ///     Any run of whitespace (<c>\s+</c>) — stripped out of a captured number so Python-style spacing such as
        ///     <c>"+ 1"</c> or <c>"-   9"</c> parses as a plain signed integer.
        /// </summary>
        /// <returns>The process-wide source-generated <see cref="Regex"/> singleton for <c>\s+</c>.</returns>
        /// <remarks>Generated rather than cached for the reason given on <see cref="SliceNotationRegex"/>.</remarks>
        [GeneratedRegex(@"\s+")]
        private static partial Regex WhitespaceRunRegex();

        /// <summary>
        ///     The debug rendering of a <see cref="SliceDef"/>, <c>(start&gt;&gt;step*count)</c>, as the
        ///     <see cref="SliceDef(string)"/> constructor parses it back.
        /// </summary>
        /// <returns>The process-wide source-generated <see cref="Regex"/> singleton for the rendering pattern.</returns>
        /// <remarks>
        ///     Lives on <see cref="Slice"/> (which is <c>partial</c>) because a generated regex needs a partial
        ///     declaring type and <see cref="SliceDef"/> is a public struct whose declaration stays untouched. Generated
        ///     rather than cached for the reason given on <see cref="SliceNotationRegex"/>.
        /// </remarks>
        [GeneratedRegex(@"\((\d+)>>(-?\d+)\*(\d+)\)")]
        internal static partial Regex SliceDefRenderingRegex();

        /// <summary>
        /// Creates Python array slice notation out of an array of Slice objects (mainly used for tests)
        /// </summary>
        public static string FormatSlices(params Slice[] slices)
        {
            return string.Join(",", slices.Select(s => s.ToString()));
        }

        private void Parse(string slice_notation)
        {
            if (string.IsNullOrEmpty(slice_notation))
                throw new ArgumentException("Slice notation expected, got empty string or null");
            var match = SliceNotationRegex().Match(slice_notation);
            if (!match.Success)
                throw new ArgumentException($"Invalid slice notation: '{slice_notation}'");
            if (match.Groups["ellipsis"].Success)
            {
                Start = 0;
                Stop = 0;
                Step = 1;
                IsEllipsis = true;
                return;
            }
            if (match.Groups["newaxis"].Success)
            {
                Start = 0;
                Stop = 0;
                Step = 1;
                IsNewAxis = true;
                return;
            }
            if (match.Groups["index"].Success)
            {
                if (!long.TryParse(WhitespaceRunRegex().Replace(match.Groups["index"].Value ?? "", ""), out var start))
                    throw new ArgumentException($"Invalid value for index: '{match.Groups["index"].Value}'");
                Start = start;
                Stop = start + 1;
                Step = 1; // special case for dimensionality reduction by picking a single element
                IsIndex = true;
                return;
            }
            var start_string = WhitespaceRunRegex().Replace(match.Groups["start"].Value ?? "", ""); // removing spaces from match to be able to parse what python allows, like: "+ 1" or  "-   9";
            var stop_string = WhitespaceRunRegex().Replace(match.Groups["stop"].Value ?? "", "");
            var step_string = WhitespaceRunRegex().Replace(match.Groups["step"].Value ?? "", "");

            if (string.IsNullOrWhiteSpace(start_string))
                Start = null;
            else
            {
                if (!long.TryParse(start_string, out var start))
                    throw new ArgumentException($"Invalid value for start: {start_string}");
                Start = start;
            }

            if (string.IsNullOrWhiteSpace(stop_string))
                Stop = null;
            else
            {
                if (!long.TryParse(stop_string, out var stop))
                    throw new ArgumentException($"Invalid value for stop: {stop_string}");
                Stop = stop;
            }

            if (string.IsNullOrWhiteSpace(step_string))
                Step = 1;
            else
            {
                if (!long.TryParse(step_string, out var step))
                    throw new ArgumentException($"Invalid value for step: {step_string}");
                Step = step;
            }
        }

        #region Equality comparison

        public static bool operator ==(Slice a, Slice b)
        {
            if (ReferenceEquals(a, b))
                return true;

            if (a is null || b is null)
                return false;

            return a.Start == b.Start && a.Stop == b.Stop && a.Step == b.Step;
        }

        public static bool operator !=(Slice a, Slice b)
        {
            return !(a == b);
        }

        public override bool Equals(object obj)
        {
            if (obj == null)
                return false;

            if (obj.GetType() != typeof(Slice))
                return false;

            var b = (Slice)obj;
            return Start == b.Start && Stop == b.Stop && Step == b.Step;
        }

        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }

        #endregion

        public override string ToString()
        {
            if (IsIndex)
                return $"{Start ?? 0}";
            else if (IsNewAxis)
                return "np.newaxis";
            else if (IsEllipsis)
                return "...";
            var optional_step = Step == 1 ? "" : $":{Step}";
            return $"{(Start == 0 ? "" : Start.ToString())}:{(Stop == null ? "" : Stop.ToString())}{optional_step}";
        }

        // return the size of the slice, given the data dimension on this axis
        // note: this works only with sanitized shapes!
        [MethodImpl(Inline)]
        public long GetSize()
        {
            var astep = Math.Abs(Step);
            return (Math.Abs(Start.Value - Stop.Value) + (astep - 1)) / astep;
        }

        /// <summary>
        /// Converts the user Slice into an internal SliceDef which is easier to calculate with
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        [MethodImpl(OptimizeAndInline)]
        public SliceDef ToSliceDef(long dim)
        {
            if (IsIndex)
            {
                var index = Start ?? 0;
                if (index < 0)
                {
                    if (Math.Abs(index) > dim)
                        throw new ArgumentException($"Index {index} is out of bounds for the axis with size {dim}");
                    return new SliceDef(dim + index);
                }

                if (index > 0 && index >= dim)
                    throw new ArgumentException($"Index {index} is out of bounds for the axis with size {dim}");
                return new SliceDef(index);
            }

            if (Step == 0)
                return new SliceDef() {Count = 0, Start = 0, Step = 0};
            var astep = Math.Abs(Step);
            if (Step > 0)
            {
                var start = Start ?? 0;
                var stop = Stop ?? dim;
                if (start >= dim)
                    return new SliceDef() {Count = 0, Start = 0, Step = 0};
                if (start < 0)
                    start = Math.Abs(start) <= dim ? dim + start : 0;
                if (stop > dim)
                    stop = dim;
                if (stop < 0)
                    stop = Math.Abs(stop) <= dim ? dim + stop : 0;
                if (start >= stop)
                    return new SliceDef() {Count = 0, Start = 0, Step = 0};
                var count = (Math.Abs(start - stop) + (astep - 1)) / astep;
                return new SliceDef() {Start = start, Step = Step, Count = count};
            }
            else
            {
                // negative step!
                var start = Start ?? dim - 1;
                var stop = Stop ?? -1;
                if (start < 0)
                    // A negative start more negative than -dim clamps to -1 ("before the beginning"
                    // when walking backwards) -> empty slice, NOT 0. NumPy: arange(3)[-7::-2] == [].
                    // (Clamping to 0 wrongly yielded a length-1 slice starting at index 0.)
                    start = Math.Abs(start) <= dim ? dim + start : -1;
                if (start >= dim)
                    start = dim - 1;
                if (Stop < 0)
                    stop = Math.Abs(stop) <= dim ? dim + stop : -1;
                if (start <= stop)
                    return new SliceDef() {Count = 0, Start = 0, Step = 0};
                var count = (Math.Abs(start - stop) + (astep - 1)) / astep;
                var retval = new SliceDef() {Start = start, Step = Step, Count = count};
                return retval;
            }
        }

        /// <summary>
        /// Backwards-compatible overload accepting int dimension.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public SliceDef ToSliceDef(int dim) => ToSliceDef((long)dim);


        #region Operators

        public static Slice operator ++(Slice a)
        {
            if (a.Start.HasValue)
                a.Start++;
            if (a.Stop.HasValue)
                a.Stop++;
            return a;
        }

        public static Slice operator --(Slice a)
        {
            if (a.Start.HasValue)
                a.Start--;
            if (a.Stop.HasValue)
                a.Stop--;
            return a;
        }

        public static implicit operator Slice(long index) => Slice.Index(index);
        public static implicit operator Slice(int index) => Slice.Index(index);
        public static implicit operator Slice(string slice) => new Slice(slice);
        //public static implicit operator Slice(NDArray selection) => Slice.Select(selection);

        #endregion
    }

    public struct SliceDef
    {
        public long Start; // start index in array
        public long Step; // positive => forward from Start,
        public long Count; // number of steps to take from Start (1 means just take Start, 0 means take nothing, -1 means this is an index)

        public SliceDef(long start, long step, long count)
        {
            (Start, Step, Count) = (start, step, count);
        }

        public SliceDef(long idx)
        {
            (Start, Step, Count) = (idx, 1, -1);
        }

        /// <summary>
        /// Backwards-compatible constructor accepting int parameters.
        /// </summary>
        public SliceDef(int start, int step, int count) : this((long)start, (long)step, (long)count)
        {
        }

        /// <summary>
        /// Backwards-compatible constructor accepting int index.
        /// </summary>
        public SliceDef(int idx) : this((long)idx)
        {
        }

        /// <summary>
        /// (Start>>Step*Count)
        /// </summary>
        /// <param name="def"></param>
        public SliceDef(string def)
        {
            if (def == "()")
            {
                (Start, Step, Count) = (0, 0, 0);
                return;
            }

            var m = Slice.SliceDefRenderingRegex().Match(def);
            Start = long.Parse(m.Groups[1].Value);
            Step = long.Parse(m.Groups[2].Value);
            Count = long.Parse(m.Groups[3].Value);
        }

        public bool IsIndex
        {
            [MethodImpl(OptimizeAndInline)] get => Count == -1;
        }

        /// <summary>
        /// reverts the order of the slice sequence
        /// </summary>
        /// <returns></returns>
        [MethodImpl(OptimizeAndInline)]
        public SliceDef Invert()
        {
            return new SliceDef() {Count = Count, Start = (Start + Step * Count), Step = -Step};
        }

        public override string ToString()
        {
            if (IsIndex)
                return $"[{Start}]";
            if (Count <= 0)
                return "()";
            return $"({Start}>>{Step}*{Count})";
        }

        /// <summary>
        /// Merge calculates the resulting one-time slice on the original data if it is sliced repeatedly
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public SliceDef Merge(SliceDef other)
        {
            if (other.Count == 0)
                return new SliceDef() {Start = 0, Step = 0, Count = 0};
            var self = this;
            if (other.IsIndex)
                return new SliceDef(self.Start + other.Start * self.Step);
            var result = new SliceDef() {Start = self.Start + other.Start * self.Step, Step = Step * other.Step, Count = other.Count,};
            return result;
        }
    }
}
