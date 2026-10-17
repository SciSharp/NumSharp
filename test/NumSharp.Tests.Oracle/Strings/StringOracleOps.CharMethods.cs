using System;
using System.Collections.Generic;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strcharmeth</c> (the <c>numpy.char</c> plan's C4 / C5): the chararray
    ///     string members (<c>cha_call</c>) and the <c>numpy.char</c> string functions (<c>cha_fn</c>, names
    ///     <c>char.*</c>) — numpy.char's own <c>partition</c> / <c>rpartition</c> and the functions it re-exports from
    ///     numpy.strings. Each call binds the C# member a ported NumPy line binds: the positional arguments in NumPy's
    ///     order, a keyword argument by its NumPy name.
    /// </summary>
    internal static partial class StringOracleOps
    {
        /// <summary>
        ///     One argument of a section-<c>strcharmeth</c> call: positional argument <paramref name="index"/> when the case
        ///     passes that many, else the keyword <paramref name="name"/>, else null (NumPy's default: the C# members read a
        ///     null optional argument as the default).
        /// </summary>
        /// <param name="a">The positional arguments.</param>
        /// <param name="index">The parameter's position.</param>
        /// <param name="args">The case's argument builder.</param>
        /// <param name="p">The case parameters (the keywords).</param>
        /// <param name="name">The parameter's NumPy name.</param>
        /// <returns>The argument, or null.</returns>
        private static object MethodArg(object[] a, int index, BehaviorArgs args, Dictionary<string, JsonElement> p, string name)
            => index < a.Length ? a[index] : Kw(args, p, name, out _);

        /// <summary>
        ///     <c>cha_call</c> for the chararray string members of section <c>strcharmeth</c>, or false when
        ///     <paramref name="name"/> is not one of them.
        /// </summary>
        /// <param name="name">The member's NumPy name.</param>
        /// <param name="c">The receiver.</param>
        /// <param name="a">The positional arguments.</param>
        /// <param name="args">The case's argument builder.</param>
        /// <param name="p">The case parameters.</param>
        /// <param name="result">The member's result.</param>
        /// <returns>True when <paramref name="name"/> is a section-<c>strcharmeth</c> member.</returns>
        private static bool TryCharStringMember(string name, NDCharArray c, object[] a, BehaviorArgs args,
                                                Dictionary<string, JsonElement> p, out object result)
        {
            object arg(int index, string kw) => MethodArg(a, index, args, p, kw);
            switch (name)
            {
                case "rpartition":
                    result = c.rpartition(arg(0, "sep"));
                    return true;
                case "center":
                    result = c.center(arg(0, "width"), arg(1, "fillchar"));
                    return true;
                case "ljust":
                    result = c.ljust(arg(0, "width"), arg(1, "fillchar"));
                    return true;
                case "rjust":
                    result = c.rjust(arg(0, "width"), arg(1, "fillchar"));
                    return true;
                case "zfill":
                    result = c.zfill(arg(0, "width"));
                    return true;
                case "expandtabs":
                    result = c.expandtabs(arg(0, "tabsize"));
                    return true;
            }
            result = null;
            return false;
        }

        /// <summary>
        ///     <c>cha_fn</c> for the <c>numpy.char</c> string functions of section <c>strcharmeth</c> (<c>char.*</c>).
        /// </summary>
        /// <param name="fn">The function's name, <c>char.</c> and NumPy's name.</param>
        /// <param name="a">The positional arguments.</param>
        /// <param name="args">The case's argument builder.</param>
        /// <param name="p">The case parameters.</param>
        /// <returns>The function's result.</returns>
        /// <exception cref="InvalidOperationException">No function of that name (a generator / replay mismatch).</exception>
        private static object CharStringFunction(string fn, object[] a, BehaviorArgs args, Dictionary<string, JsonElement> p)
        {
            object arg(int index, string kw) => MethodArg(a, index, args, p, kw);
            return fn switch
            {
                "char.partition" => np.@char.partition(arg(0, "a"), arg(1, "sep")),
                "char.rpartition" => np.@char.rpartition(arg(0, "a"), arg(1, "sep")),
                "char.center" => np.@char.center(arg(0, "a"), arg(1, "width"), arg(2, "fillchar")),
                "char.ljust" => np.@char.ljust(arg(0, "a"), arg(1, "width"), arg(2, "fillchar")),
                "char.rjust" => np.@char.rjust(arg(0, "a"), arg(1, "width"), arg(2, "fillchar")),
                "char.zfill" => np.@char.zfill(arg(0, "a"), arg(1, "width")),
                "char.expandtabs" => np.@char.expandtabs(arg(0, "a"), arg(1, "tabsize")),
                var f => throw new InvalidOperationException($"the chararray behavior sections have no function '{f}'"),
            };
        }
    }
}
