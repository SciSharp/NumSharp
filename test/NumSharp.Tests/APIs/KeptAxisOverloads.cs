using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace NumSharp.Tests.APIs
{
    /// <summary>
    ///     The owner's batch-6 decision for every member that takes NumPy's axis as <see cref="NDAxis"/>: the <c>int</c> and
    ///     <c>int[]</c> overloads stay beside it (an <c>int?</c> one for a one-integer axis: argmax / argmin / nanargmax /
    ///     nanargmin), each a one-line forward to the NDAxis member. A test that pins "one NumPy signature" counts the
    ///     overloads WITHOUT these kept forwards - the NumPy signature is the NDAxis member - so it filters them out here.
    /// </summary>
    /// <remarks>
    ///     A kept form is recognized structurally, never by a list: its axis-like parameters (<c>axis</c>, <c>axes</c>,
    ///     <c>source</c>, <c>destination</c>) are <c>int</c>, <c>int?</c> or <c>int[]</c>, and the same type declares a sibling
    ///     of the same name whose parameters are identical except that those parameters are <see cref="NDAxis"/> /
    ///     <c>NDAxis?</c>. So a stray integer overload WITHOUT such a sibling still counts, and fails a one-signature test.
    /// </remarks>
    internal static class KeptAxisOverloads
    {
        /// <summary>The parameter names that carry NumPy's axis argument.</summary>
        private static readonly HashSet<string> AxisNames = new() { "axis", "axes", "source", "destination" };

        /// <summary>
        ///     Whether <paramref name="method"/> is a kept integer forward: integer-typed axis parameters, and an NDAxis
        ///     sibling with otherwise the same parameter list.
        /// </summary>
        /// <param name="method">An overload.</param>
        /// <returns>True for a kept <c>int</c> / <c>int?</c> / <c>int[]</c> forward of an NDAxis member.</returns>
        internal static bool IsKeptIntegerForm(MethodInfo method)
        {
            var ps = method.GetParameters();
            if (!ps.Any(p => AxisNames.Contains(p.Name) && IsIntegerAxis(p.ParameterType)))
                return false;
            const BindingFlags all = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
            return method.DeclaringType!.GetMethods(all).Any(sibling =>
            {
                if (sibling == method || sibling.Name != method.Name)
                    return false;
                var qs = sibling.GetParameters();
                if (qs.Length != ps.Length)
                    return false;
                for (int i = 0; i < ps.Length; i++)
                {
                    if (ps[i].Name != qs[i].Name)
                        return false;
                    bool axisSlot = AxisNames.Contains(ps[i].Name) && IsIntegerAxis(ps[i].ParameterType);
                    if (axisSlot ? !IsNDAxis(qs[i].ParameterType) : ps[i].ParameterType != qs[i].ParameterType)
                        return false;
                }
                return true;
            });
        }

        /// <summary>The overloads without the kept integer forwards: the NumPy-signature forms (and any S8 / NumSharp form).</summary>
        /// <param name="forms">Overloads of one name.</param>
        /// <returns>The forms a one-NumPy-signature test counts.</returns>
        internal static MethodInfo[] WithoutKeptIntegerForms(this IEnumerable<MethodInfo> forms)
            => forms.Where(m => !IsKeptIntegerForm(m)).ToArray();

        /// <summary>Whether a parameter type is one of the kept integer axis types.</summary>
        /// <param name="type">The parameter type.</param>
        /// <returns>True for <c>int</c>, <c>int?</c> and <c>int[]</c>.</returns>
        private static bool IsIntegerAxis(Type type) => type == typeof(int) || type == typeof(int?) || type == typeof(int[]);

        /// <summary>Whether a parameter type is NumPy's axis type.</summary>
        /// <param name="type">The parameter type.</param>
        /// <returns>True for <see cref="NDAxis"/> and <c>NDAxis?</c>.</returns>
        private static bool IsNDAxis(Type type) => type == typeof(NDAxis) || type == typeof(NDAxis?);
    }
}
