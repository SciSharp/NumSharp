using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Tensor contraction over the given axes — a sum product over the last
        ///     <paramref name="axes"/> axes of <paramref name="a"/> and the first
        ///     <paramref name="axes"/> of <paramref name="b"/>.
        /// </summary>
        /// <param name="axes">
        ///     How many trailing axes of <c>a</c> to contract against leading axes of <c>b</c>.
        ///     <c>0</c> gives the outer (tensor) product; <c>1</c> is <see cref="dot"/>; <c>2</c>
        ///     (the default) is the double contraction. A NEGATIVE count contracts nothing — NumPy
        ///     forms <c>range(-axes, 0)</c>, which is empty for any <c>axes &lt;= 0</c>.
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html</remarks>
        public static NDArray tensordot(NDArray a, NDArray b, int axes = 2)
        {
            var axesA = new List<int>();
            var axesB = new List<int>();
            for (int k = -axes; k < 0; k++)
                axesA.Add(k);
            for (int k = 0; k < axes; k++)
                axesB.Add(k);

            return tensordot(a, b, axesA.ToArray(), axesB.ToArray());
        }

        /// <summary>
        ///     Tensor contraction pairing one axis of <paramref name="a"/> with one of
        ///     <paramref name="b"/> — NumPy's <c>axes=(int, int)</c> spelling.
        /// </summary>
        public static NDArray tensordot(NDArray a, NDArray b, (int AxisA, int AxisB) axes)
            => tensordot(a, b, new[] {axes.AxisA}, new[] {axes.AxisB});

        /// <summary>
        ///     Tensor contraction pairing <paramref name="axesA"/> of <paramref name="a"/> with
        ///     <paramref name="axesB"/> of <paramref name="b"/>, element by element — NumPy's
        ///     <c>axes=(list, list)</c> spelling.
        /// </summary>
        /// <exception cref="ValueError">
        ///     <c>"duplicate axes are not allowed in tensordot"</c> when either list repeats a raw
        ///     axis value (checked first, ahead of everything), or <c>"shape-mismatch for sum"</c> for
        ///     unequal list lengths and mismatched contracted extents alike.
        /// </exception>
        /// <exception cref="IndexError">
        ///     <c>"tuple index out of range"</c> when a contraction axis is out of range — NumPy
        ///     indexes the shape tuple with the raw axis, so this is an index error, not a shape
        ///     mismatch. A mismatched extent found earlier in the list settles as the shape mismatch
        ///     first, hiding a later out-of-range axis.
        /// </exception>
        [NDScoped]
        public static NDArray tensordot(NDArray a, NDArray b, int[] axesA, int[] axesB)
        {
            // Port of numpy/_core/numeric.py :: tensordot.
            var shapeA = a.Shape.dimensions;
            var shapeB = b.Shape.dimensions;
            int na = axesA.Length;
            int nb = axesB.Length;

            // NumPy rejects LITERAL duplicates first — `len(set(axes_a)) != len(axes_a)` — on the
            // RAW axis values, ahead of every shape check and before negatives are normalized. So
            // `([5,5],[0,1])` is a duplicate, not an out-of-range index, and `([0,-2],…)` (a repeat
            // only AFTER normalization) is NOT caught here — it slips through exactly as upstream.
            if (HasDuplicateAxis(axesA) || HasDuplicateAxis(axesB))
                throw new ValueError("duplicate axes are not allowed in tensordot");

            var la = (int[])axesA.Clone();
            var lb = (int[])axesB.Clone();

            bool equal = na == nb;
            if (equal)
            {
                for (int k = 0; k < na; k++)
                {
                    // NumPy indexes the shape TUPLE with the raw axis (`as_[axes_a[k]]`), so an
                    // out-of-range axis raises IndexError "tuple index out of range", NOT a shape
                    // mismatch. Both operands are indexed before the extents are compared and the
                    // comparison short-circuits the loop, so a mismatch at k hides an out-of-range
                    // axis at k+1 — which is why `([0,1],[0,5])` settles as a shape mismatch (a[0] !=
                    // b[0]) while `([5,0],[0,1])` is an IndexError (a[5] is reached first).
                    int ia = ResolveContractionAxis(la[k], a.ndim);
                    int ib = ResolveContractionAxis(lb[k], b.ndim);
                    if (shapeA[ia] != shapeB[ib])
                    {
                        equal = false;
                        break;
                    }

                    la[k] = ia;
                    lb[k] = ib;
                }
            }

            if (!equal)
                throw new ValueError("shape-mismatch for sum");

            // Move the contracted axes to the END of a and to the FRONT of b, collapse each side to
            // a matrix, and let the ordinary matrix product do the work.
            var keptA = Complement(a.ndim, la);
            var keptB = Complement(b.ndim, lb);

            long innerA = 1;
            foreach (int axis in la)
                innerA *= shapeA[axis];
            long outerA = 1;
            var oldA = new long[keptA.Length];
            for (int i = 0; i < keptA.Length; i++)
            {
                oldA[i] = shapeA[keptA[i]];
                outerA *= oldA[i];
            }

            long innerB = 1;
            foreach (int axis in lb)
                innerB *= shapeB[axis];
            long outerB = 1;
            var oldB = new long[keptB.Length];
            for (int i = 0; i < keptB.Length; i++)
            {
                oldB[i] = shapeB[keptB[i]];
                outerB *= oldB[i];
            }

            var at = reshape(transpose(a, Concat(keptA, la)), new[] {outerA, innerA});
            var bt = reshape(transpose(b, Concat(lb, keptB)), new[] {innerB, outerB});

            var product = dot(at, bt);

            var resultShape = new long[oldA.Length + oldB.Length];
            oldA.CopyTo(resultShape, 0);
            oldB.CopyTo(resultShape, oldA.Length);
            return reshape(product, resultShape);
        }

        /// <summary>NumPy's <c>len(set(axes)) != len(axes)</c> — a literal repeat among the raw axis values.</summary>
        private static bool HasDuplicateAxis(int[] axes)
        {
            for (int i = 0; i < axes.Length; i++)
            for (int j = i + 1; j < axes.Length; j++)
            {
                if (axes[i] == axes[j])
                    return true;
            }

            return false;
        }

        /// <summary>
        ///     Resolves one contraction axis the way NumPy's <c>shape[axis]</c> does: a negative index
        ///     wraps, and anything still out of range is an <see cref="IndexError"/>
        ///     "tuple index out of range" — never a shape mismatch.
        /// </summary>
        private static int ResolveContractionAxis(int axis, int ndim)
        {
            int resolved = axis < 0 ? axis + ndim : axis;
            if (resolved < 0 || resolved >= ndim)
                throw new IndexError("tuple index out of range");
            return resolved;
        }

        /// <summary>The axes of a rank-<paramref name="ndim"/> array not named in <paramref name="taken"/>, in order.</summary>
        private static int[] Complement(int ndim, int[] taken)
        {
            var kept = new List<int>(ndim);
            for (int k = 0; k < ndim; k++)
            {
                bool found = false;
                foreach (int t in taken)
                {
                    if (t != k)
                        continue;
                    found = true;
                    break;
                }

                if (!found)
                    kept.Add(k);
            }

            return kept.ToArray();
        }

        private static int[] Concat(int[] first, int[] second)
        {
            var joined = new int[first.Length + second.Length];
            first.CopyTo(joined, 0);
            second.CopyTo(joined, first.Length);
            return joined;
        }
    }
}
