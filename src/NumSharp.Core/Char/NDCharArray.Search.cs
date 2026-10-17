// =============================================================================
// NDCharArray.Search.cs — chararray.count / find / rfind / index / rindex / startswith / endswith
// (numpy/_core/defchararray.py)
// =============================================================================
//
// NumPy's members are one line each over the numpy.char functions (numpy.strings' own), with the chararray as the text:
//     count(self, sub, start=0, end=None)         -> count(self, sub, start, end)
//     find / rfind / index / rindex               -> the same shape
//     startswith(self, prefix, start=0, end=None) -> startswith(self, prefix, start, end)
//     endswith(self, suffix, start=0, end=None)   -> endswith(self, suffix, start, end)
// The function reads the stored text UNSTRIPPED (a ufunc reads the array, never chararray.__getitem__: 'ab  '.count(' ')
// is 2 although c[0] reads 'ab') and returns its int64 / bool result as is: chararray.__array_wrap__ keeps the class only
// for a text result, so every member result is a PLAIN array, a 0-d one for a 0-d chararray (NumPy's wrap does not make
// a scalar of it). There is no asarray step and no chararray view to avoid, so the members call the function directly.
// =============================================================================

namespace NumSharp
{
    public partial class NDCharArray
    {
        /// <summary>
        ///     NumPy's <c>chararray.count(sub, start=0, end=None)</c>: the number of non-overlapping occurrences of
        ///     <paramref name="sub"/> in every element's <c>[start:end]</c> range — the stored text, unstripped, so trailing
        ///     whitespace is searched too.
        /// </summary>
        /// <param name="sub">The substrings, broadcast against the elements: this array's class (a <c>bytes</c> substring for
        ///     a <c>U</c> chararray, a <c>str</c> one for an <c>S</c> chararray or a number has no loop).</param>
        /// <param name="start">The range starts in slice notation (negative counts from the end), any integer dtype,
        ///     broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c> (the whole
        ///     string).</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes (NumPy's <c>ufunc 'count' did not contain a loop with
        ///     signature matching types ...</c>).</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast (a float or
        ///     text: <c>Cannot cast ufunc 'count' input 2 from dtype('float64') to dtype('int64') with casting rule
        ///     'same_kind'</c>).</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <remarks>NumPy's member is <c>count(self, sub, start, end)</c>; see <see cref="CharModule.count"/> for the
        ///     function.</remarks>
        public NDArray count(object sub, object start = null, object end = null) => np.@char.count(this, sub, start, end);

        /// <summary>
        ///     NumPy's <c>chararray.find(sub, start=0, end=None)</c>: for every element, the lowest index where
        ///     <paramref name="sub"/> occurs inside its <c>[start:end]</c> range, or -1 (the stored text, unstripped).
        /// </summary>
        /// <param name="sub">The substrings, broadcast against the elements (this array's class).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        public NDArray find(object sub, object start = null, object end = null) => np.@char.find(this, sub, start, end);

        /// <summary>
        ///     NumPy's <c>chararray.rfind(sub, start=0, end=None)</c>: for every element, the highest index where
        ///     <paramref name="sub"/> occurs inside its <c>[start:end]</c> range, or -1 (the stored text, unstripped).
        /// </summary>
        /// <param name="sub">The substrings, broadcast against the elements (this array's class).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        public NDArray rfind(object sub, object start = null, object end = null) => np.@char.rfind(this, sub, start, end);

        /// <summary>
        ///     NumPy's <c>chararray.index(sub, start=0, end=None)</c>: <see cref="find"/>, but an element with no match
        ///     raises <c>ValueError: substring not found</c>.
        /// </summary>
        /// <param name="sub">The substrings, broadcast against the elements (this array's class).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <exception cref="ValueError">An element without a match (<c>substring not found</c>).</exception>
        public NDArray index(object sub, object start = null, object end = null) => np.@char.index(this, sub, start, end);

        /// <summary>
        ///     NumPy's <c>chararray.rindex(sub, start=0, end=None)</c>: <see cref="rfind"/>, but an element with no match
        ///     raises <c>ValueError: substring not found</c>.
        /// </summary>
        /// <param name="sub">The substrings, broadcast against the elements (this array's class).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain int64 array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="sub"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        /// <exception cref="ValueError">An element without a match (<c>substring not found</c>).</exception>
        public NDArray rindex(object sub, object start = null, object end = null) => np.@char.rindex(this, sub, start, end);

        /// <summary>
        ///     NumPy's <c>chararray.startswith(prefix, start=0, end=None)</c>: whether every element's <c>[start:end]</c>
        ///     range begins with <paramref name="prefix"/> (the stored text, unstripped).
        /// </summary>
        /// <param name="prefix">The prefixes, broadcast against the elements (this array's class).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain bool array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="prefix"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        public NDArray startswith(object prefix, object start = null, object end = null)
            => np.@char.startswith(this, prefix, start, end);

        /// <summary>
        ///     NumPy's <c>chararray.endswith(suffix, start=0, end=None)</c>: whether every element's <c>[start:end]</c> range
        ///     ends with <paramref name="suffix"/> — the stored text, unstripped, so <c>endswith(' ')</c> is True for an
        ///     element stored with trailing spaces.
        /// </summary>
        /// <param name="suffix">The suffixes, broadcast against the elements (this array's class).</param>
        /// <param name="start">The range starts in slice notation, broadcast; null is NumPy's default 0.</param>
        /// <param name="end">The range ends in slice notation, broadcast; null is NumPy's <c>None</c>.</param>
        /// <returns>A plain bool array of the broadcast shape (0-d for a 0-d chararray).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="suffix"/> is null.</exception>
        /// <exception cref="TypeError">No loop for the classes.</exception>
        /// <exception cref="System.ArgumentException">A start / end that cannot reach int64 by a same_kind cast.</exception>
        /// <exception cref="System.OverflowException">A start / end literal past int64.</exception>
        /// <exception cref="IncorrectShapeException">An argument does not broadcast.</exception>
        public NDArray endswith(object suffix, object start = null, object end = null)
            => np.@char.endswith(this, suffix, start, end);
    }
}
