// NumPy's set routines np.union1d, np.setxor1d and np.setdiff1d (numpy/lib/_arraysetops_impl.py) as DefaultEngine's
// members, NumPy's statements as they are (a surviving NaN keeps the bits np.unique / np.sort leave it: NumPy 2.5's sort
// keeps every NaN's bits, where NumPy 2.4's rewrote it as the positive quiet NaN). NumPy's statements call np.unique,
// np.concatenate, np.ravel, np.sort, np.isin, np.isnan, np.any and np.where; here they are this engine's or the
// operand's engine members (np.unique's Unique with its defaults, values only) and the array object's (the ravel, the
// concatenation, the one-element flag, NDArray.where).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped]
        public override NDArray Union1d(NDArray ar1, NDArray ar2)
        {
            // Text: NumPy's statement below runs unchanged — the concatenation formats numbers into text and decodes bytes
            // beside str, and np.unique sorts the text.
            // Boundary scope: the ravel views, the concatenate buffer and unique's internals are
            // reclaimed at exit; only the final array is yielded.
            // NumPy: unique(concatenate((ar1, ar2), axis=None)) — axis=None flattens both operands.
            return Unique(
                NDArray.concatenate(new[] { NDArray.ravel(ar1, 'C'), NDArray.ravel(ar2, 'C') }, 0, null, null, "same_kind"),
                false, false, false, null, true, true)[0];
        }

        /// <inheritdoc />
        [NDScoped]
        public override NDArray Setxor1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            // Text runs the statements below unchanged: np.unique and np.sort order the text, the concatenation formats
            // numbers and decodes bytes as NumPy's does, and the text != builds the flags.
            // Boundary scope: the unique/ravel/concatenate/sort temps and the flag masks are
            // reclaimed at exit; only the surviving-values array is yielded.

            if (!assume_unique)
            {
                ar1 = Unique(ar1, false, false, false, null, true, true)[0];
                ar2 = Unique(ar2, false, false, false, null, true, true)[0];
            }

            // concatenate((ar1, ar2), axis=None) — flattens both.
            NDArray aux = NDArray.concatenate(new[] { NDArray.ravel(ar1, 'C'), NDArray.ravel(ar2, 'C') }, 0, null, null, "same_kind");
            if (aux.size == 0)
                return aux;

            aux = Sort(aux, -1, null, null, null);

            // flag = concatenate(([True], aux[1:] != aux[:-1], [True]))
            NDArray trueOne = new NDArray(Unmanaged.ArraySlice.FromArray(new[] { true }, true), Shape.Vector(1)) { TensorEngine = this };   // np.array([True])
            NDArray neq = aux["1:"] != aux[":-1"];
            NDArray flag = NDArray.concatenate(new[] { trueOne, neq, trueOne }, 0, null, null, "same_kind");

            // return aux[flag[1:] & flag[:-1]]  — values appearing exactly once survive
            NDArray sel = flag["1:"] & flag[":-1"];
            return aux[sel];
        }

        /// <inheritdoc />
        [NDScoped]
        public override NDArray Setdiff1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            // Text runs the statements below unchanged: np.unique sorts the text and np.isin runs NumPy's _isin over it.
            // Boundary scope: the unique/ravel temps and the isin membership mask are reclaimed
            // at exit; only the selected-values array is yielded.

            if (assume_unique)
            {
                ar1 = NDArray.ravel(ar1, 'C');
            }
            else
            {
                ar1 = Unique(ar1, false, false, false, null, true, true)[0];
                ar2 = Unique(ar2, false, false, false, null, true, true)[0];
            }

            // ar1[isin(ar1, ar2, assume_unique=True, invert=True)] — keep only the values absent from ar2.
            NDArray mask = Isin(ar1, ar2, true, true, null);   // np.isin(ar1, ar2, assume_unique=True, invert=True)
            return ar1[mask];
        }
    }
}
