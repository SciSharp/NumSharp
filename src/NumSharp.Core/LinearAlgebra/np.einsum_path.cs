using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluates the lowest-cost contraction order for an <see cref="einsum(string, NDArray[])"/>
        ///     expression, considering the creation of intermediate arrays.
        /// </summary>
        /// <param name="subscripts">The einsum subscripts, e.g. <c>"ij,jk,kl-&gt;il"</c>.</param>
        /// <param name="operands">The arrays the subscripts label — only their SHAPES are read.</param>
        /// <returns>
        ///     The <see cref="EinsumPath"/> (NumPy's <c>['einsum_path', …]</c> list) and a printable
        ///     representation of the path.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html
        ///     <para>
        ///     A route-for-route port of NumPy 2.4.2's <c>numpy.einsum_path</c> (its greedy/optimal
        ///     planner in <c>einsumfunc.py</c>). The <see cref="EinsumPath"/> and every numeric metric
        ///     in the string are byte-identical to NumPy; the only difference is the placeholder letters
        ///     an <c>...</c> expands to in the printed string, which NumPy itself does not pin (they are
        ///     drawn from a hash-randomized set and vary per process). NumSharp draws them
        ///     deterministically, so the path and the numbers always match.
        ///     </para>
        ///     <para>
        ///     Spell the operands either as an <c>NDArray[]</c> (this overload) or variadically through
        ///     <see cref="einsum_path(object[])"/> — <c>np.einsum_path("ij,jk", a, b)</c>. The default
        ///     is <c>optimize: "greedy"</c>, NumPy's default for <c>einsum_path</c> (note
        ///     <see cref="einsum(string, NDArray[])"/>'s own default is <c>false</c>).
        ///     </para>
        /// </remarks>
        public static (EinsumPath path, string repr) einsum_path(string subscripts, NDArray[] operands)
            => EinsumPathCore(subscripts, operands, "greedy");

        /// <summary>
        ///     Evaluates the contraction order, choosing the path type via <paramref name="optimize"/>.
        /// </summary>
        /// <param name="optimize">
        ///     <c>false</c>/<c>null</c> (no optimization), <c>true</c> (≙ <c>"greedy"</c>),
        ///     <c>"greedy"</c>, <c>"optimal"</c>, a precomputed <see cref="EinsumPath"/> (an explicit
        ///     path), or a <c>("greedy"|"optimal", maxIntermediateSize)</c> tuple that caps the largest
        ///     intermediate. Anything else is rejected exactly as NumPy rejects it.
        /// </param>
        /// <inheritdoc cref="einsum_path(string, NDArray[])"/>
        public static (EinsumPath path, string repr) einsum_path(string subscripts, NDArray[] operands, object optimize)
            => EinsumPathCore(subscripts, operands, optimize);

        /// <summary>
        ///     Evaluates the contraction order variadically — <c>np.einsum_path("ij,jk", a, b)</c> — and
        ///     in NumPy's SUBLIST spelling — <c>np.einsum_path(a, [0,1], b, [1,2], [0,2])</c>.
        ///     Optimization is <c>"greedy"</c>; use the <c>NDArray[]</c> overload to choose a different
        ///     <c>optimize</c>.
        /// </summary>
        /// <inheritdoc cref="einsum_path(string, NDArray[])"/>
        public static (EinsumPath path, string repr) einsum_path(params object[] operands)
        {
            if (operands is null || operands.Length == 0)
                throw new ValueError(
                    "must specify the einstein sum subscripts string and at least one operand, " +
                    "or at least one operand and its corresponding subscripts list");

            // A leading string is the ordinary spelling reached through this overload.
            if (operands[0] is string subscripts)
            {
                var arrays = new NDArray[operands.Length - 1];
                for (int i = 1; i < operands.Length; i++)
                {
                    arrays[i - 1] = operands[i] as NDArray
                                    ?? throw new TypeError(
                                        $"einsum_path operand {i - 1} must be an NDArray, got {operands[i]?.GetType().Name ?? "null"}");
                }

                return EinsumPathCore(subscripts, arrays, "greedy");
            }

            string rendered = EinsumSubscripts.FromSublists(operands, out NDArray[] parsed);
            return EinsumPathCore(rendered, parsed, "greedy");
        }

        private static (EinsumPath path, string repr) EinsumPathCore(string subscripts, NDArray[] operands, object optimize)
        {
            if (subscripts is null)
                throw new ArgumentNullException(nameof(subscripts));

            operands ??= Array.Empty<NDArray>();
            long[][] shapes = ExtractShapes(operands);
            EinsumPathPlanner.Directive directive = ResolveOptimize(optimize);
            var (path, repr) = EinsumPathPlanner.Compute(subscripts, shapes, directive);
            return (new EinsumPath(path), repr);
        }

        private static long[][] ExtractShapes(NDArray[] operands)
        {
            var shapes = new long[operands.Length][];
            for (int i = 0; i < operands.Length; i++)
            {
                if (operands[i] is null)
                    throw new ValueError($"einsum_path operand {i} is null");
                shapes[i] = operands[i].Shape.dimensions;
            }

            return shapes;
        }

        /// <summary>
        ///     Port of <c>einsum_path</c>'s <c>path_type</c> resolution — maps NumPy's Python-typed
        ///     <c>optimize</c> onto C# types, in NumPy's exact decision order.
        /// </summary>
        private static EinsumPathPlanner.Directive ResolveOptimize(object optimize)
        {
            // path_type = optimize; True -> 'greedy'; None -> False.
            object pathType = optimize;
            if (pathType is bool b)
                pathType = b ? "greedy" : (object)false;
            if (pathType is null)
                pathType = false;

            // (path_type is False) or isinstance(path_type, str): leave it to the planner.
            if (pathType is bool bf && bf == false)
                return new EinsumPathPlanner.Directive(noOpt: true, algorithm: null, explicitPath: null, memoryLimit: null);
            if (pathType is string s)
                return new EinsumPathPlanner.Directive(noOpt: false, algorithm: s, explicitPath: null, memoryLimit: null);

            // An explicit path — our own EinsumPath, or a list/array whose head is the marker.
            if (pathType is EinsumPath ep)
                return new EinsumPathPlanner.Directive(false, null, StepsOf(ep), null);
            if (TryExplicitPath(pathType, out int[][] explicitSteps))
                return new EinsumPathPlanner.Directive(false, null, explicitSteps, null);

            // Path tuple with memory limit: (str, int|float).
            if (pathType is ITuple tuple && tuple.Length == 2 && tuple[0] is string algo && IsNumber(tuple[1]))
                return new EinsumPathPlanner.Directive(false, algo, null, ToLong(tuple[1]));
            if (pathType is object[] arr && arr.Length == 2 && arr[0] is string arrAlgo && IsNumber(arr[1]))
                return new EinsumPathPlanner.Directive(false, arrAlgo, null, ToLong(arr[1]));

            // A bare number leaks Python's len() error; anything else is "Did not understand".
            if (IsNumber(pathType))
                throw new TypeError($"object of type '{PyTypeName(pathType)}' has no len()");

            throw new TypeError($"Did not understand the path: {PyRepr(pathType)}");
        }

        private static int[][] StepsOf(EinsumPath path)
        {
            var steps = new int[path.Count][];
            for (int i = 0; i < path.Count; i++)
                steps[i] = path[i];
            return steps;
        }

        /// <summary>
        ///     Recognises a raw list/array explicit path — one whose first element is the marker
        ///     string <c>"einsum_path"</c> — and pulls out the contraction tuples (NumPy's
        ///     <c>path_type[1:]</c>). An empty tail is legal here and surfaces later as NumPy's
        ///     "Invalid einsum_path" RuntimeError, exactly as upstream.
        /// </summary>
        private static bool TryExplicitPath(object pathType, out int[][] steps)
        {
            steps = null;
            IReadOnlyList<object> items = pathType switch
            {
                object[] arr => arr,
                IReadOnlyList<object> list => list,
                _ => null
            };

            if (items is null || items.Count == 0 || !(items[0] is string marker) || marker != EinsumPath.Marker)
                return false;

            var result = new int[items.Count - 1][];
            for (int i = 1; i < items.Count; i++)
                result[i - 1] = ToIntTuple(items[i]);
            steps = result;
            return true;
        }

        private static int[] ToIntTuple(object entry)
        {
            switch (entry)
            {
                case int[] ints:
                    return ints;
                case ITuple tuple:
                {
                    var r = new int[tuple.Length];
                    for (int i = 0; i < tuple.Length; i++)
                        r[i] = Convert.ToInt32(tuple[i], CultureInfo.InvariantCulture);
                    return r;
                }
                case IEnumerable seq when entry is not string:
                {
                    var list = new List<int>();
                    foreach (object o in seq)
                        list.Add(Convert.ToInt32(o, CultureInfo.InvariantCulture));
                    return list.ToArray();
                }
                default:
                    throw new TypeError($"Did not understand the path: {PyRepr(entry)}");
            }
        }

        private static bool IsNumber(object o) =>
            o is int or long or short or byte or sbyte or uint or ulong or ushort or float or double or decimal;

        private static long ToLong(object o) => Convert.ToInt64(o, CultureInfo.InvariantCulture);

        private static string PyTypeName(object o) => o switch
        {
            float or double or decimal => "float",
            bool => "bool",
            _ => "int"
        };

        /// <summary>A best-effort Python-style repr for the "Did not understand the path" message.</summary>
        private static string PyRepr(object o)
        {
            switch (o)
            {
                case null:
                    return "None";
                case string s:
                    return "'" + s + "'";
                case bool b:
                    return b ? "True" : "False";
                case ITuple tuple:
                {
                    var sb = new StringBuilder("(");
                    for (int i = 0; i < tuple.Length; i++)
                    {
                        if (i > 0)
                            sb.Append(", ");
                        sb.Append(PyRepr(tuple[i]));
                    }

                    if (tuple.Length == 1)
                        sb.Append(',');
                    return sb.Append(')').ToString();
                }
                case object[] arr:
                {
                    var sb = new StringBuilder("[");
                    for (int i = 0; i < arr.Length; i++)
                    {
                        if (i > 0)
                            sb.Append(", ");
                        sb.Append(PyRepr(arr[i]));
                    }

                    return sb.Append(']').ToString();
                }
                default:
                    return Convert.ToString(o, CultureInfo.InvariantCulture);
            }
        }
    }
}
