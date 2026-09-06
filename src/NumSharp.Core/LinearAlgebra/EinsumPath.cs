using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace NumSharp
{
    /// <summary>
    ///     The contraction path returned by <see cref="np.einsum_path(string, NDArray[])"/> — NumPy's
    ///     <c>['einsum_path', (1, 2), (0, 1)]</c> list expressed as a value.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     This stands in for the FIRST element of NumPy's <c>einsum_path</c> return tuple: a Python
    ///     list whose head is the string marker <c>'einsum_path'</c> and whose tail is one integer
    ///     tuple per contraction step, each naming the operand positions contracted at that step. The
    ///     head marker is implicit here — <see cref="Steps"/> and the indexer expose the contraction
    ///     tuples directly, and <see cref="ToString"/> renders the full NumPy list (marker included).
    ///     </para>
    ///     <para>
    ///     A value of this type round-trips into <c>optimize:</c> on both
    ///     <see cref="np.einsum_path(string, NDArray[], object)"/> and
    ///     <see cref="np.einsum(string, NDArray[], NDArray, NPTypeCode?, char, string, object)"/>,
    ///     exactly as NumPy accepts <c>path_info[0]</c> back as an explicit path.
    ///     </para>
    /// </remarks>
    public readonly struct EinsumPath : IEquatable<EinsumPath>, IReadOnlyList<int[]>
    {
        /// <summary>NumPy's list marker, <c>path[0]</c>.</summary>
        public const string Marker = "einsum_path";

        private readonly int[][] _steps;

        /// <param name="steps">One entry per contraction step; each is the operand positions contracted.</param>
        public EinsumPath(int[][] steps) => _steps = steps ?? Array.Empty<int[]>();

        /// <summary>The contraction steps — NumPy's <c>path[1:]</c> (the marker excluded).</summary>
        public IReadOnlyList<int[]> Steps => _steps ?? Array.Empty<int[]>();

        /// <summary>The number of contraction steps (the marker is NOT counted).</summary>
        public int Count => _steps?.Length ?? 0;

        /// <summary>The <paramref name="index"/>-th contraction step (the marker is NOT indexed).</summary>
        public int[] this[int index] => (_steps ?? Array.Empty<int[]>())[index];

        /// <summary>
        ///     NumPy's list form: <c>{"einsum_path", int[]{1,2}, int[]{0,1}}</c> — the marker string
        ///     followed by each contraction tuple, for callers that want the raw heterogeneous list.
        /// </summary>
        public object[] ToList()
        {
            var steps = _steps ?? Array.Empty<int[]>();
            var list = new object[steps.Length + 1];
            list[0] = Marker;
            for (int i = 0; i < steps.Length; i++)
                list[i + 1] = steps[i];
            return list;
        }

        public IEnumerator<int[]> GetEnumerator()
        {
            foreach (var step in _steps ?? Array.Empty<int[]>())
                yield return step;
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public bool Equals(EinsumPath other)
        {
            var a = _steps ?? Array.Empty<int[]>();
            var b = other._steps ?? Array.Empty<int[]>();
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i].Length != b[i].Length)
                    return false;
                for (int j = 0; j < a[i].Length; j++)
                {
                    if (a[i][j] != b[i][j])
                        return false;
                }
            }

            return true;
        }

        public override bool Equals(object obj) => obj is EinsumPath other && Equals(other);

        public override int GetHashCode()
        {
            var hash = new HashCode();
            foreach (var step in _steps ?? Array.Empty<int[]>())
            {
                foreach (int v in step)
                    hash.Add(v);
                hash.Add(-1); // step boundary
            }

            return hash.ToHashCode();
        }

        /// <summary>Renders NumPy's list, e.g. <c>['einsum_path', (1, 2), (0, 1)]</c>.</summary>
        public override string ToString()
        {
            var sb = new StringBuilder("['").Append(Marker).Append('\'');
            foreach (var step in _steps ?? Array.Empty<int[]>())
            {
                sb.Append(", (");
                for (int i = 0; i < step.Length; i++)
                {
                    if (i > 0)
                        sb.Append(", ");
                    sb.Append(step[i]);
                }

                if (step.Length == 1)
                    sb.Append(','); // Python 1-tuple prints (0,)
                sb.Append(')');
            }

            return sb.Append(']').ToString();
        }
    }
}
