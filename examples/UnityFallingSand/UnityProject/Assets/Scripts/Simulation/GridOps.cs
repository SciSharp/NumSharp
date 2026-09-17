using System;
using NumSharp;

namespace NumSharp.Examples.FallingSand.Simulation
{
    /// <summary>
    /// The vectorized grid primitives every falling-sand rule is built from: neighbour-shifts and the
    /// conflict-free swap. Keeping them here — as whole-array NumSharp operations — is what lets the
    /// cellular automaton update the entire grid with a handful of array calls per rule instead of a
    /// per-cell double loop.
    /// </summary>
    /// <remarks>
    /// All operations take the grid dimensions from the array itself, so they are independent of any
    /// world size. Out-of-grid neighbours are filled with a caller-chosen value (a wall for materials, a
    /// "not moving" false for masks), which makes the four grid edges behave like solid boundaries with
    /// no special-casing in the rules.
    /// </remarks>
    public static class GridOps
    {
        /// <summary>
        /// Returns the neighbour field <c>result[r,c] = a[r+dr, c+dc]</c> for an INTEGER grid, with
        /// out-of-grid positions set to <paramref name="fill"/>. This is how a rule reads "the cell below"
        /// (<c>dr=+1</c>), "the cell to the left" (<c>dc=-1</c>), a diagonal, etc., for the whole grid at
        /// once.
        /// </summary>
        /// <param name="a">The source grid, a 2-D int array.</param>
        /// <param name="dr">Row offset of the neighbour (−1, 0 or +1).</param>
        /// <param name="dc">Column offset of the neighbour (−1, 0 or +1).</param>
        /// <param name="fill">Value for positions whose neighbour is off the grid (use a wall id for materials).</param>
        /// <returns>A new array of the same shape holding the shifted values.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        public static NDArray Shift(NDArray a, int dr, int dc, int fill)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            int h = (int)a.shape[0], w = (int)a.shape[1];
            var res = np.full(new Shape(h, w), fill, NPTypeCode.Int32);
            CopyShifted(a, res, dr, dc, h, w);
            return res;
        }

        /// <summary>
        /// Returns the neighbour field <c>result[r,c] = m[r+dr, c+dc]</c> for a BOOLEAN mask, with
        /// out-of-grid positions set to false (an off-grid neighbour is never "moving" / never "open").
        /// </summary>
        /// <param name="m">The source mask, a 2-D bool array.</param>
        /// <param name="dr">Row offset of the neighbour (−1, 0 or +1).</param>
        /// <param name="dc">Column offset of the neighbour (−1, 0 or +1).</param>
        /// <returns>A new bool array of the same shape holding the shifted mask.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="m"/> is null.</exception>
        public static NDArray ShiftMask(NDArray m, int dr, int dc)
        {
            if (m is null) throw new ArgumentNullException(nameof(m));
            int h = (int)m.shape[0], w = (int)m.shape[1];
            var res = np.zeros(new Shape(h, w), NPTypeCode.Boolean);   // false = off-grid default
            CopyShifted(m, res, dr, dc, h, w);
            return res;
        }

        /// <summary>
        /// Copies the overlapping window of <paramref name="src"/> into <paramref name="dst"/> so that
        /// <c>dst[r,c] = src[r+dr, c+dc]</c>, leaving the non-overlapping edge at its pre-filled value. The
        /// slice ranges are computed once per axis, so the whole shift is a single vectorized block copy.
        /// </summary>
        /// <param name="src">Source array.</param>
        /// <param name="dst">Pre-filled destination array (same shape).</param>
        /// <param name="dr">Row offset.</param>
        /// <param name="dc">Column offset.</param>
        /// <param name="h">Grid height.</param>
        /// <param name="w">Grid width.</param>
        private static void CopyShifted(NDArray src, NDArray dst, int dr, int dc, int h, int w)
        {
            (string dstR, string srcR) = RangeFor(dr, h);
            (string dstC, string srcC) = RangeFor(dc, w);
            // Assign the shifted block via slice ranges; the untouched edge keeps dst's fill value.
            dst[$"{dstR}, {dstC}"] = src[$"{srcR}, {srcC}"];
        }

        /// <summary>
        /// Computes the destination and source slice strings along one axis for a neighbour offset. For
        /// <c>d=+1</c> we want <c>dst[r]=src[r+1]</c>, i.e. destination rows <c>0..L-2</c> take source rows
        /// <c>1..L-1</c>; for <c>d=-1</c> it is the mirror; for <c>d=0</c> the whole axis.
        /// </summary>
        /// <param name="d">Offset along the axis (−1, 0 or +1).</param>
        /// <param name="length">The axis length.</param>
        /// <returns>The destination and source slice strings for that axis.</returns>
        private static (string dst, string src) RangeFor(int d, int length)
        {
            if (d == 0) return ($"0:{length}", $"0:{length}");
            if (d == 1) return ($"0:{length - 1}", $"1:{length}");    // dst[r] = src[r+1]
            return ($"1:{length}", $"0:{length - 1}");               // dst[r] = src[r-1]
        }

        /// <summary>
        /// Applies a set of same-direction moves as a conflict-free, MASS-CONSERVING swap: every cell whose
        /// <paramref name="move"/> flag is set exchanges contents with its <c>(dr,dc)</c> neighbour. This
        /// single primitive implements falling, sinking, sliding and spreading — the direction and the
        /// selection mask are all that differ between rules.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>Why it cannot duplicate or drop material.</b> Within one direction, each target cell has
        /// exactly one possible source (the neighbour offset by −(dr,dc)), so two movers can never land on
        /// the same cell. The one remaining hazard is a "chain" — a cell that is both a mover and the
        /// target of another mover. It is removed by the guard <c>allowed = move AND NOT move(neighbour)</c>:
        /// a cell may move only if its target is not ALSO moving. This provably makes each firing pair
        /// disjoint, so the update is a permutation of the grid and every material count is preserved
        /// exactly (verified over long random runs by the sample's harness).
        /// </para>
        /// <para>
        /// A chain therefore advances one cell per step, which is exactly how gravity should propagate in a
        /// cellular automaton (the bottom particle falls first, then the one above it next step).
        /// </para>
        /// </remarks>
        /// <param name="cells">The material grid.</param>
        /// <param name="move">A bool mask: cell wants to swap with its <c>(dr,dc)</c> neighbour.</param>
        /// <param name="dr">Row offset of the swap direction.</param>
        /// <param name="dc">Column offset of the swap direction.</param>
        /// <param name="fillMat">The material used for off-grid neighbours (a wall id) — irrelevant to the result but required by the shifts.</param>
        /// <returns>A new grid with all allowed swaps applied.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="cells"/> or <paramref name="move"/> is null.</exception>
        public static NDArray ApplySwap(NDArray cells, NDArray move, int dr, int dc, int fillMat)
        {
            if (cells is null) throw new ArgumentNullException(nameof(cells));
            if (move is null) throw new ArgumentNullException(nameof(move));

            // A mover may fire only if its target isn't itself moving the same way (the anti-chain guard).
            var allowed = move & !ShiftMask(move, dr, dc);
            // A cell RECEIVES if the cell at −(dr,dc) is an allowed mover pointing at it.
            var recv = ShiftMask(allowed, -dr, -dc);
            var targetMat = Shift(cells, dr, dc, fillMat);     // material this mover swaps toward
            var sourceMat = Shift(cells, -dr, -dc, fillMat);   // material a receiver swaps in
            // allowed and recv are disjoint (proven), so the nested where is unambiguous: a mover takes its
            // target's content, a receiver takes its source's content, everything else is unchanged.
            return np.where(allowed, targetMat, np.where(recv, sourceMat, cells));
        }
    }
}
