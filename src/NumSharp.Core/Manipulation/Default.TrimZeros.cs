// NumPy's np.trim_zeros (numpy/lib/_function_base_impl.py) as DefaultEngine's member: its statements as the facade held
// them - the trim spec's check, normalize_axis_tuple, the no-axis return, the text dtypes' truth probe and the bounding
// box of the nonzero hyperplanes, one trimmed axis at a time. The file sits beside its facade while it calls np.*:
// ratchet R2 counts np.* calls against every file under Backends/ (the np-layer dtype plan, C3); it moves under
// Backends/Default/ in P6.

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // the truth probe of a text filt, np.any's masks and np.nonzero's indices are this call's own: only the view returned leaves
        public override NDArray TrimZeros(NDArray filt, string trim = "fb", int[] axis = null)
        {
            // NumPy lower-cases and validates the trim spec against the four accepted spellings.
            trim = (trim ?? string.Empty).ToLowerInvariant();
            if (trim != "fb" && trim != "bf" && trim != "f" && trim != "b")
                throw new ValueError($"unexpected character(s) in `trim`: '{trim}'");

            int ndim = filt.ndim;

            // Resolve the axis tuple. axis == null means "all axes" (bounding box over the whole array);
            // an explicit list is normalized (negative wrap + out-of-range check) and must not repeat an axis.
            var trimAxis = new bool[ndim];
            bool anyAxis;
            if (axis == null)
            {
                for (int i = 0; i < ndim; i++)
                    trimAxis[i] = true;
                anyAxis = ndim > 0;
            }
            else
            {
                anyAxis = axis.Length > 0;
                // normalize_axis_tuple(axis, ndim, argname="axis"): every axis range-checked first ("axis: axis 3 is out
                // of bounds ..."), then the "repeated axis in `axis` argument" check — NumPy's order, so a tuple holding
                // both a repeat and an out-of-range axis reports the latter.
                foreach (int a in DefaultEngine.normalize_axis_tuple(axis, ndim, "axis"))
                    trimAxis[a] = true;
            }

            // No trimming requested (axis == () or a 0-d input) -> return the input unmodified, like NumPy.
            if (!anyAxis)
                return filt;

            // Text: NumPy's argwhere reads the strings' truth values through the dtype's nonzero function — after the trim
            // and axis checks and the no-axis return above. The bounding box is found on that truth array (the probe), and
            // the slices below cut `filt` itself, so the result stays a view of the text. (The probe is reclaimed with the
            // scope; np.any over the truth array cannot raise, where np.any over the text would take the cast route.)
            var probe = Backends.Strings.NDStringUfuncs.IsText(filt)
                ? Backends.Strings.NDStringTruth.Truth(filt, Backends.Strings.NDStringTruth.Flavor.Nonzero)
                : filt;

            bool trimFront = trim.IndexOf('f') >= 0;
            bool trimBack = trim.IndexOf('b') >= 0;

            // The bounding box of the non-zero elements is SEPARABLE per dimension: the first/last non-zero
            // coordinate along axis d equals the first/last index whose perpendicular hyperplane contains any
            // non-zero value. So rather than materializing every non-zero coordinate (NumPy's argwhere.min/max
            // over the whole array), reduce "any non-zero" over all OTHER axes down to a 1-D mask and take the
            // first/last of its non-zero positions. Only the axes actually being trimmed are projected.
            var start = new long?[ndim]; // null => open slice edge (Slice.All for untrimmed axes)
            var stop = new long?[ndim];

            var others = new int[ndim > 0 ? ndim - 1 : 0];
            for (int d = 0; d < ndim; d++)
            {
                if (!trimAxis[d])
                    continue; // untrimmed axis keeps null/null -> Slice.All

                // positions: sorted indices of the non-zero hyperplanes along axis d.
                //   1-D  -> the non-zero element indices directly (skip the projection).
                //   N-D  -> np.any over every other axis gives a 1-D mask; its non-zero indices are the edges.
                NDArray positions;
                if (ndim == 1)
                {
                    positions = np.nonzero(probe)[0];
                }
                else
                {
                    int oi = 0;
                    for (int k = 0; k < ndim; k++)
                        if (k != d)
                            others[oi++] = k;
                    positions = np.nonzero(np.any(probe, others))[0];
                }

                long n = positions.size;
                if (n == 0)
                {
                    // All-zero (or empty) along d -> empty slice; trim spec ignored, matching NumPy (start == stop).
                    start[d] = 0L;
                    stop[d] = 0L;
                }
                else
                {
                    long first = (long)positions[0];         // ascending -> ends are the first/last non-zero
                    long last = (long)positions[(int)n - 1];
                    start[d] = trimFront ? first : (long?)null; // 'f' absent -> keep the front (open start)
                    stop[d] = trimBack ? last + 1 : (long?)null; // 'b' absent -> keep the back  (open stop)
                }
            }

            // Trimmed axes are sliced to [start, stop); untouched axes span the full dimension.
            var slices = new Slice[ndim];
            for (int ax = 0; ax < ndim; ax++)
                slices[ax] = trimAxis[ax] ? new Slice(start[ax], stop[ax]) : Slice.All;

            return filt[slices];
        }
    }
}
