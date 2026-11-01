// NumPy's np.tensordot (numpy/_core/numeric.py) as DefaultEngine's members: its statements, the integer axes form
// building NumPy's two axis lists and the list form contracting through one dot product. The facades are
// LinearAlgebra/np.tensordot.cs; numpy.linalg's Array-API tensordot (DefaultLinalgEngine.ArrayApi.cs) calls them too.

using System.Collections.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        public override NDArray Tensordot(NDArray a, NDArray b, int axes = 2)
        {
            // NumPy's `axes_a = list(range(-axes, 0))`, `axes_b = list(range(0, axes))`: a count of 0 or less leaves both
            // lists empty (the outer product). Text reaches the final dot after the axis checks, and dot answers for it.
            var axesA = new List<int>();
            var axesB = new List<int>();
            for (int k = -axes; k < 0; k++)
                axesA.Add(k);
            for (int k = 0; k < axes; k++)
                axesB.Add(k);

            return Tensordot(a, b, axesA.ToArray(), axesB.ToArray());
        }

        /// <inheritdoc />
        [NDScoped /* the transposed operands, their matrices and the product are reclaimed; the result view is yielded */]
        public override NDArray Tensordot(NDArray a, NDArray b, int[] axesA, int[] axesB)
        {
            // NumPy's statements; text runs every axis check below, then the final dot (the product of the two reshaped
            // matrices) gives NumPy's answer for it.
            var shapeA = a.Shape.dimensions;
            var shapeB = b.Shape.dimensions;
            int na = axesA.Length;
            int nb = axesB.Length;

            // NumPy rejects LITERAL duplicates first - `len(set(axes_a)) != len(axes_a)` - on the RAW axis values, ahead of
            // every shape check and before negatives are normalized. So `([5,5],[0,1])` is a duplicate, not an out-of-range
            // index, and `([0,-2],...)` (a repeat only AFTER normalization) is NOT caught here: it slips through as upstream.
            if (HasDuplicateAxis(axesA) || HasDuplicateAxis(axesB))
                throw new ValueError("duplicate axes are not allowed in tensordot");

            var la = (int[])axesA.Clone();
            var lb = (int[])axesB.Clone();

            bool equal = na == nb;
            if (equal)
            {
                for (int k = 0; k < na; k++)
                {
                    // NumPy indexes the shape TUPLE with the raw axis (`as_[axes_a[k]]`), so an out-of-range axis raises
                    // IndexError "tuple index out of range", NOT a shape mismatch. Both operands are indexed before the
                    // extents are compared and the comparison short-circuits the loop, so a mismatch at k hides an
                    // out-of-range axis at k+1 - which is why `([0,1],[0,5])` settles as a shape mismatch (a[0] != b[0])
                    // while `([5,0],[0,1])` is an IndexError (a[5] is reached first).
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

            // Move the contracted axes to the END of a and to the FRONT of b, collapse each side to a matrix, and let the
            // ordinary matrix product do the work.
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

            // np.transpose's member on each operand's engine, then np.reshape's body (the array object's reshape), then
            // np.dot's member on the left matrix's engine (its OpenBLAS route included).
            NDArray transposedA = a.TensorEngine.Transpose(a, Concat(keptA, la));
            NDArray at = transposedA.reshape(new[] { outerA, innerA });
            NDArray transposedB = b.TensorEngine.Transpose(b, Concat(lb, keptB));
            NDArray bt = transposedB.reshape(new[] { innerB, outerB });

            NDArray product = at.TensorEngine.Dot(at, bt, null);

            var resultShape = new long[oldA.Length + oldB.Length];
            oldA.CopyTo(resultShape, 0);
            oldB.CopyTo(resultShape, oldA.Length);
            return product.reshape(resultShape);
        }

        /// <summary>NumPy's <c>len(set(axes)) != len(axes)</c>: a literal repeat among the raw axis values, tested before any
        ///     axis is normalized.</summary>
        /// <param name="axes">The raw axis list.</param>
        /// <returns>True when two entries are equal.</returns>
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
        ///     Resolves one contraction axis the way NumPy's <c>shape[axis]</c> does: a negative index wraps once, and anything
        ///     still out of range is an <see cref="IndexError"/> "tuple index out of range" - never a shape mismatch.
        /// </summary>
        /// <param name="axis">The raw axis.</param>
        /// <param name="ndim">The operand's number of dimensions.</param>
        /// <returns>The axis in [0, ndim).</returns>
        /// <exception cref="IndexError">The axis is outside [-ndim, ndim).</exception>
        private static int ResolveContractionAxis(int axis, int ndim)
        {
            int resolved = axis < 0 ? axis + ndim : axis;
            if (resolved < 0 || resolved >= ndim)
                throw new IndexError("tuple index out of range");
            return resolved;
        }

        /// <summary>The axes of a rank-<paramref name="ndim"/> array not named in <paramref name="taken"/>, in order (NumPy's
        ///     <c>notin</c> list).</summary>
        /// <param name="ndim">The operand's number of dimensions.</param>
        /// <param name="taken">The normalized contracted axes.</param>
        /// <returns>The kept axes, ascending.</returns>
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

        /// <summary>Two axis lists joined, the transpose order <c>notin + axes_a</c> / <c>axes_b + notin</c>.</summary>
        /// <param name="first">The leading axes.</param>
        /// <param name="second">The trailing axes.</param>
        /// <returns>A new array holding <paramref name="first"/> then <paramref name="second"/>.</returns>
        private static int[] Concat(int[] first, int[] second)
        {
            var joined = new int[first.Length + second.Length];
            first.CopyTo(joined, 0);
            second.CopyTo(joined, first.Length);
            return joined;
        }
    }
}
