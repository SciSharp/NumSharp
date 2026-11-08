// NumPy's np.sort_complex (numpy/lib/_function_base_impl.py) as DefaultEngine's member: its two statements, the sort in
// the input's dtype (the engine's Sort, np.sort's member) and the conversion to complex.

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // the sorted copy is yielded, or reclaimed when the conversion to complex replaces it
        public override NDArray SortComplex(NDArray a)
        {
            // Text runs NumPy's two statements as they are: b.sort() orders the text along the last axis (a 0-d input
            // refuses the axis), and astype('D') parses each string into a complex number with Python's complex() rules.
            // np.sort == array(a, copy=True) + in-place sort along the default last axis; the sort
            // runs in the ORIGINAL dtype (an int64 past 2^53 must order by its exact value, not by
            // its double image), and only the sorted result is widened.
            var b = Sort(a, -1);
            return b.typecode == NPTypeCode.Complex ? b : b.astype(NPTypeCode.Complex);
        }
    }
}
