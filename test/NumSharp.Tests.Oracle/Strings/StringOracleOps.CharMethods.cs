using System;
using System.Collections.Generic;
using System.Text.Json;

namespace NumSharp.Tests.Strings
{
    /// <summary>
    ///     The string oracle's ops for section <c>strcharmeth</c> (the <c>numpy.char</c> plan's C4 / C5): the chararray
    ///     string members (<c>cha_call</c>) and the <c>numpy.char</c> string functions (<c>cha_fn</c>, names
    ///     <c>char.*</c>) — numpy.char's own <c>partition</c> / <c>rpartition</c> and the functions it re-exports from
    ///     numpy.strings (the predicates among them are numpy.strings' ufuncs themselves). Each call binds the C#
    ///     member a ported NumPy line binds: the positional arguments in NumPy's order, a keyword argument by its NumPy
    ///     name.
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
                case "count":
                    result = c.count(arg(0, "sub"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "find":
                    result = c.find(arg(0, "sub"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "rfind":
                    result = c.rfind(arg(0, "sub"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "index":
                    result = c.index(arg(0, "sub"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "rindex":
                    result = c.rindex(arg(0, "sub"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "startswith":
                    result = c.startswith(arg(0, "prefix"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "endswith":
                    result = c.endswith(arg(0, "suffix"), arg(1, "start"), arg(2, "end"));
                    return true;
                case "strip":
                    result = c.strip(arg(0, "chars"));
                    return true;
                case "lstrip":
                    result = c.lstrip(arg(0, "chars"));
                    return true;
                case "rstrip":
                    result = c.rstrip(arg(0, "chars"));
                    return true;
                case "replace":
                    result = c.replace(arg(0, "old"), arg(1, "new"), arg(2, "count"));
                    return true;
                case "isalnum":
                    result = c.isalnum();
                    return true;
                case "isalpha":
                    result = c.isalpha();
                    return true;
                case "isdecimal":
                    result = c.isdecimal();
                    return true;
                case "isdigit":
                    result = c.isdigit();
                    return true;
                case "islower":
                    result = c.islower();
                    return true;
                case "isnumeric":
                    result = c.isnumeric();
                    return true;
                case "isspace":
                    result = c.isspace();
                    return true;
                case "istitle":
                    result = c.istitle();
                    return true;
                case "isupper":
                    result = c.isupper();
                    return true;
                case "capitalize":
                    result = c.capitalize();
                    return true;
                case "lower":
                    result = c.lower();
                    return true;
                case "swapcase":
                    result = c.swapcase();
                    return true;
                case "title":
                    result = c.title();
                    return true;
                case "upper":
                    result = c.upper();
                    return true;
                case "translate":
                    result = c.translate(arg(0, "table"), arg(1, "deletechars"));
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
                // str_len is a ufunc: out= may come positionally, where= and dtype= by keyword.
                "char.str_len" => np.@char.str_len(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.count" => np.@char.count(arg(0, "a"), arg(1, "sub"), arg(2, "start"), arg(3, "end")),
                "char.find" => np.@char.find(arg(0, "a"), arg(1, "sub"), arg(2, "start"), arg(3, "end")),
                "char.rfind" => np.@char.rfind(arg(0, "a"), arg(1, "sub"), arg(2, "start"), arg(3, "end")),
                "char.index" => np.@char.index(arg(0, "a"), arg(1, "sub"), arg(2, "start"), arg(3, "end")),
                "char.rindex" => np.@char.rindex(arg(0, "a"), arg(1, "sub"), arg(2, "start"), arg(3, "end")),
                "char.startswith" => np.@char.startswith(arg(0, "a"), arg(1, "prefix"), arg(2, "start"), arg(3, "end")),
                "char.endswith" => np.@char.endswith(arg(0, "a"), arg(1, "suffix"), arg(2, "start"), arg(3, "end")),
                "char.strip" => np.@char.strip(arg(0, "a"), arg(1, "chars")),
                "char.lstrip" => np.@char.lstrip(arg(0, "a"), arg(1, "chars")),
                "char.rstrip" => np.@char.rstrip(arg(0, "a"), arg(1, "chars")),
                "char.replace" => np.@char.replace(arg(0, "a"), arg(1, "old"), arg(2, "new"), arg(3, "count")),
                // The predicates are numpy.strings' ufuncs: out= may come positionally, where= and dtype= by keyword.
                "char.isalnum" => np.@char.isalnum(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.isalpha" => np.@char.isalpha(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.isdecimal" => np.@char.isdecimal(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                       (DType)arg(3, "dtype")),
                "char.isdigit" => np.@char.isdigit(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.islower" => np.@char.islower(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.isnumeric" => np.@char.isnumeric(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                       (DType)arg(3, "dtype")),
                "char.isspace" => np.@char.isspace(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.istitle" => np.@char.istitle(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                "char.isupper" => np.@char.isupper(arg(0, "x"), (NDArray)arg(1, "out"), (NDArray)arg(2, "where"),
                                                   (DType)arg(3, "dtype")),
                // The case maps and translate: np.asarray(a) and _vec_string, not ufuncs.
                "char.capitalize" => np.@char.capitalize(arg(0, "a")),
                "char.lower" => np.@char.lower(arg(0, "a")),
                "char.swapcase" => np.@char.swapcase(arg(0, "a")),
                "char.title" => np.@char.title(arg(0, "a")),
                "char.upper" => np.@char.upper(arg(0, "a")),
                "char.translate" => np.@char.translate(arg(0, "a"), arg(1, "table"), arg(2, "deletechars")),
                // mod: np.asarray(a) and _vec_string with str.__mod__ / bytes.__mod__ per element (numpy.strings' own).
                "char.mod" => np.@char.mod(arg(0, "a"), arg(1, "values")),
                var f => throw new InvalidOperationException($"the chararray behavior sections have no function '{f}'"),
            };
        }
    }
}
