using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Fill the main diagonal of the given array of any dimensionality — <b>in place</b>.
        /// </summary>
        /// <param name="a">Array whose diagonal is to be filled; it is modified in place.</param>
        /// <param name="val">
        ///     Value(s) to write into the diagonal. A scalar is repeated; a sequence is raveled
        ///     and then <b>tiled cyclically</b> (and truncated) to the diagonal's length. <c>null</c> is
        ///     Python's <c>None</c>: NaN (NumPy's positive <c>NPY_NAN</c>) for a float array, <c>nan+nanj</c>
        ///     for complex, <c>False</c> for bool, <c>'None'</c> (truncated) for <c>S</c> / <c>U</c>, a
        ///     <c>StringDType</c>'s NA object or <c>'None'</c>.
        /// </param>
        /// <param name="wrap">
        ///     For tall matrices the diagonal "wraps" after N columns and continues on the row
        ///     below. Off by default.
        /// </param>
        /// <exception cref="ArgumentException">
        ///     <c>array must be at least 2-d</c>, <c>All dimensions of input must be of equal length</c>
        ///     (ndim &gt; 2 with a non-hyper-cubic shape), or <c>underlying array is read-only</c>
        ///     — all verbatim NumPy <c>ValueError</c> texts.
        /// </exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="val"/> is null (<c>None</c>) and <paramref name="a"/> is an integer (or
        ///     <c>decimal</c>) array — Python's <c>int(None)</c>, raised even for an empty diagonal.</exception>
        /// <exception cref="ValueError">A <c>StringDType(coerce=False)</c> array given a value that is not text
        ///     (<c>None</c> included).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html
        ///     <para>
        ///     <b>Values tile, they do not broadcast.</b> NumPy's body ends in
        ///     <c>a.flat[:end:step] = val</c>, and flat-iterator assignment repeats a short
        ///     right-hand side rather than raising: probed on a 6×6, <c>[1,2,3,4]</c> fills the
        ///     diagonal as <c>[1,2,3,4,1,2]</c>, while an over-long value list is truncated and
        ///     an <b>empty</b> one is a silent no-op. <see cref="resize(NDArray,int[])"/> already
        ///     implements exactly that cyclic-tile-or-truncate rule, so it supplies the values.
        ///     </para>
        ///     <para>
        ///     <b>No element loop.</b> NumPy addresses the diagonal as a strided slice of the
        ///     flat iterator; NumSharp resolves the same positions into <em>real strides</em> and
        ///     writes through ordinary aliased views. Without wrapping the targets are
        ///     <c>(i, i, …, i)</c>, i.e. one constant stride of <c>Σ strides</c> — a single view.
        ///     With wrapping on a tall matrix the flat position of target <c>i = q·cols + r</c>
        ///     is <c>q·(cols+1)·s0 + r·(s0+s1)</c>, so the run splits into <c>ceil(count/cols)</c>
        ///     equally-strided blocks — a handful of views, still no per-element addressing.
        ///     Because the targets are computed from strides rather than from memory order, this
        ///     works unchanged on transposed, sliced and otherwise non-contiguous arrays, which
        ///     it writes through exactly as NumPy does.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static void fill_diagonal(NDArray a, object val, bool wrap = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (a is null) throw new ArgumentNullException(nameof(a));

            // NumPy's numpy.fill_diagonal: the statements are the array object's (NDArray.IndexTricks.cs)
            NDArray.FillDiagonal(a, val, wrap);
        }
    }
}