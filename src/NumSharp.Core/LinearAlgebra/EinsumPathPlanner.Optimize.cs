// np.einsum_path's optimize= resolution (NumPy's path_type branches in numpy/_core/einsumfunc.py), part of the einsum path
// planner: np.einsum_path resolves its optimize argument here, and DefaultEngine.Einsum runs the same resolution for any
// optimize= but False, as NumPy's einsum hands it to einsum_path. It lived on np (np.einsum_path.cs) until the engine's
// call to it, an up-call into np, moved it here (the np-layer dtype plan, P6).

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;

namespace NumSharp
{
    internal static partial class EinsumPathPlanner
    {
        /// <summary>
        ///     Port of <c>einsum_path</c>'s <c>path_type</c> resolution — maps NumPy's Python-typed
        ///     <c>optimize</c> onto C# types, in NumPy's exact decision order. <c>np.einsum</c> runs it too, for any
        ///     <c>optimize</c> but <c>False</c>, as NumPy's <c>einsum</c> calls <c>einsum_path</c>.
        /// </summary>
        /// <param name="optimize">NumPy's <c>optimize</c>: null (None) or a bool, a string (a path algorithm), an
        ///     <see cref="EinsumPath"/> or a list headed <c>'einsum_path'</c> (an explicit path), or a
        ///     <c>(string, number)</c> tuple or two-element <c>object[]</c> (an algorithm with a memory limit; a bool
        ///     limit is Python's int).</param>
        /// <returns>The planner directive: no optimization, an algorithm, an explicit path, or an algorithm with a
        ///     memory limit.</returns>
        /// <exception cref="TypeError">A number (<c>object of type 'int' has no len()</c>), or anything else NumPy's
        ///     branches do not take (<c>Did not understand the path: ...</c>, with a Python repr of it).</exception>
        internal static EinsumPathPlanner.Directive ResolveOptimize(object optimize)
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
                return new EinsumPathPlanner.Directive(false, algo, null, ToBigInteger(tuple[1]));
            if (pathType is object[] arr && arr.Length == 2 && arr[0] is string arrAlgo && IsNumber(arr[1]))
                return new EinsumPathPlanner.Directive(false, arrAlgo, null, ToBigInteger(arr[1]));

            // A bare number leaks Python's len() error; anything else is "Did not understand".
            if (IsNumber(pathType))
                throw new TypeError($"object of type '{PyTypeName(pathType)}' has no len()");

            throw new TypeError($"Did not understand the path: {PyRepr(pathType)}");
        }

        /// <summary>
        ///     The contraction steps of a precomputed <see cref="EinsumPath"/> in the planner's form, one operand-position
        ///     array per step. The arrays are the path's own (its indexer returns them), not copies: the planner only reads
        ///     them.
        /// </summary>
        /// <param name="path">The path an earlier <c>np.einsum_path</c> returned (its marker is not a step).</param>
        /// <returns>The steps, in the path's order.</returns>
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
        /// <param name="pathType">The resolved <c>optimize</c> value: an <c>object[]</c> or any
        ///     <see cref="IReadOnlyList{T}"/> of objects is examined, anything else is no explicit path.</param>
        /// <param name="steps">The contraction tuples after the marker, or null when this is not an explicit path.</param>
        /// <returns>True when <paramref name="pathType"/> is a list headed by <see cref="EinsumPath.Marker"/>.</returns>
        /// <exception cref="TypeError">A step after the marker is neither an <c>int[]</c>, a tuple nor a sequence
        ///     (<see cref="ToIntTuple"/>'s refusal).</exception>
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

        /// <summary>
        ///     One step of a raw explicit path as operand positions: NumPy's contraction tuple, spelled in C# as an
        ///     <c>int[]</c> (returned as it is), a tuple or any other sequence (each item converted to an <c>int</c>).
        /// </summary>
        /// <param name="entry">The step: an <c>int[]</c>, an <see cref="ITuple"/>, or an <see cref="IEnumerable"/> that is
        ///     not a string.</param>
        /// <returns>The operand positions of the step.</returns>
        /// <exception cref="TypeError">Any other value (NumPy's <c>Did not understand the path: ...</c>).</exception>
        /// <exception cref="InvalidCastException">An item that does not convert to an <c>int</c>.</exception>
        /// <exception cref="FormatException">A string item that is not an integer.</exception>
        /// <exception cref="OverflowException">An item outside the <c>int</c> range.</exception>
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

        /// <summary>
        ///     Whether a boxed value is a C# number that stands for NumPy's <c>(int, float)</c> in the
        ///     <c>optimize=(name, memory_limit)</c> tuple: any integer or floating type, and a bool, which NumPy's
        ///     <c>isinstance(x, (int, float))</c> takes for an int.
        /// </summary>
        /// <param name="o">The tuple's second item.</param>
        /// <returns>True for <c>bool</c>, <c>int</c>, <c>long</c>, <c>short</c>, <c>byte</c>, <c>sbyte</c>, <c>uint</c>,
        ///     <c>ulong</c>, <c>ushort</c>, <c>float</c>, <c>double</c> or <c>decimal</c>.</returns>
        private static bool IsNumber(object o) =>
            o is bool or int or long or short or byte or sbyte or uint or ulong or ushort or float or double or decimal;

        /// <summary>
        ///     Converts a numeric <c>optimize=("greedy", N)</c> memory limit to <see cref="BigInteger"/>.
        ///     Integer spellings are exact (and may exceed <c>long</c>); float/double/decimal spellings
        ///     truncate toward zero — matching how NumPy's <c>new_size &gt; memory_limit</c> compares an
        ///     exact bignum size against a float limit.
        /// </summary>
        /// <param name="o">The limit: a value <see cref="IsNumber"/> accepts.</param>
        /// <returns>The limit as an integer (a bool is 0 or 1, Python's <c>int(True)</c>).</returns>
        /// <exception cref="InvalidCastException">A value <see cref="IsNumber"/> refuses (the caller checks first).</exception>
        /// <exception cref="OverflowException">A float or double infinity or NaN, which no integer holds.</exception>
        private static BigInteger ToBigInteger(object o) => o switch
        {
            bool flag => flag ? 1 : 0,   // Python's int(True)
            int i => i,
            long l => l,
            short s => s,
            byte b => b,
            sbyte sb => sb,
            uint u => u,
            ulong ul => ul,
            ushort us => us,
            float f => new BigInteger(f),
            double d => new BigInteger(d),
            decimal m => new BigInteger(m),
            _ => throw new InvalidCastException($"cannot convert {o?.GetType().Name ?? "null"} to a memory limit")
        };

        /// <summary>
        ///     The Python type name of a number given as <c>optimize=</c>, for NumPy's <c>object of type '...' has no len()</c>
        ///     text: <c>float</c> for a floating type, <c>bool</c> for a bool, <c>int</c> for every integer type.
        /// </summary>
        /// <param name="o">A value <see cref="IsNumber"/> accepts.</param>
        /// <returns>The Python type name.</returns>
        private static string PyTypeName(object o) => o switch
        {
            float or double or decimal => "float",
            bool => "bool",
            _ => "int"
        };

        /// <summary>A best-effort Python-style repr for the "Did not understand the path" message.</summary>
        /// <param name="o">The value to render: None, a str, a bool, a tuple, an <c>object[]</c> list (both rendered
        ///     item by item), or anything else through its invariant-culture string.</param>
        /// <returns>The rendering, e.g. <c>None</c>, <c>'greedy'</c>, <c>True</c>, <c>('greedy', 100)</c>,
        ///     <c>[1, 2]</c>.</returns>
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
