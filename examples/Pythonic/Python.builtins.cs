using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using NumSharp;
using Python.Runtime;

// The classes of the tree carry Python's names, and those are lower case. The compiler warns that a
// lower-case type name can become a keyword of a later C# version (CS8981); the names stay as Python has them.
#pragma warning disable CS8981

namespace Pythonic;

static partial class Python
{
    /// <summary>
    ///     Python's built-in functions that the ports use: <see cref="print()"/>, <see cref="str"/> and
    ///     <see cref="format"/>. Python needs no import for them; C# needs
    ///     <c>using static Python.builtins;</c>.
    /// </summary>
    internal static class builtins
    {
        /// <summary>The module. It belongs to the process: it is not disposed.</summary>
        private static PyObject Module => PythonRuntime.Import("builtins");

        /// <summary><c>print()</c>: writes an empty line.</summary>
        public static void print() => Console.WriteLine();

        /// <summary>
        ///     <c>print(f"...")</c>: writes one line of text whose holes are formatted as Python formats
        ///     them. Write the C# string with <c>$</c> where Python has <c>f</c>, and keep the format
        ///     specs of the Python program:
        ///     <code>
        ///     print(f"accuracy   {accuracy:.4f}")        # Python
        ///     print($"accuracy   {accuracy:.4f}");       // C#
        ///     </code>
        /// </summary>
        /// <param name="text">The interpolated string. The compiler builds it through <see cref="fstring"/>, hole by hole.</param>
        /// <remarks>This overload is taken only when the <c>$"..."</c> is written directly as the argument. A string that was built before the call was formatted by .NET, with .NET's format strings.</remarks>
        public static void print(fstring text) => Console.WriteLine(text.ToString());

        /// <summary><c>print(value)</c>: writes <see cref="str"/> of one value and a line end.</summary>
        /// <param name="value">A text, a number, a NumSharp array, an object of the tree, or <c>null</c> for Python's <c>None</c>.</param>
        /// <remarks>It takes one value. Python's <c>print(a, b)</c> is <c>print($"{a} {b}")</c> here, so that each hole is formatted by Python.</remarks>
        /// <exception cref="NotSupportedException">The type of <paramref name="value"/> has no Python form in this tree.</exception>
        public static void print(object? value) => Console.WriteLine(str(value));

        /// <summary><c>str(value)</c>: the text that Python prints for a value.</summary>
        /// <param name="value">
        ///     <c>null</c> gives <c>None</c>. A <see cref="bool"/> gives <c>True</c> or <c>False</c>. A
        ///     NumSharp array is printed by NumSharp, which prints an array as NumPy does. A
        ///     <see cref="double"/> and an object of the tree are printed by Python itself, so <c>1.0</c>
        ///     stays <c>1.0</c> and <c>0.00001</c> is <c>1e-05</c>.
        /// </param>
        /// <returns>The text.</returns>
        /// <exception cref="NotSupportedException">The type of <paramref name="value"/> has no Python form in this tree.</exception>
        /// <exception cref="PythonInteropException">Python must write the value, and no usable Python was found.</exception>
        public static string str(object? value)
        {
            switch (value)
            {
                case null:
                    return "None";
                case string text:
                    return text;
                case bool flag:
                    return flag ? "True" : "False";
                case sbyte or byte or short or ushort or int or uint or long or ulong:
                    return ((IFormattable)value).ToString(null, CultureInfo.InvariantCulture);     // the digits are the same in both languages
                case NDArray array:
                    return array.ToString();
            }

            using Py.GILState gil = PythonRuntime.GIL();
            using PyObject text_py = Call(Module, "str", [value]);
            return text_py.As<string>();
        }

        /// <summary><c>format(value, format_spec)</c>: a value written with one of Python's format specs, for example <c>.4f</c>, <c>,</c>, <c>&gt;10</c> or <c>.1%</c>.</summary>
        /// <param name="value">A number, a text, a 0-d NumSharp array (NumPy formats its one element), or an object of the tree.</param>
        /// <param name="format_spec">The spec, as it stands after the colon in a Python f-string. An empty spec gives <see cref="str"/>.</param>
        /// <returns>The text.</returns>
        /// <exception cref="PythonException">Python refuses the spec for this value, for example <c>.4f</c> for a text.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="value"/> is <c>null</c> and the spec is not empty.</exception>
        /// <exception cref="NotSupportedException">The type of <paramref name="value"/> has no Python form in this tree.</exception>
        public static string format(object? value, string format_spec)
        {
            if (format_spec.Length == 0)
                return str(value);                           // Python defines format(x, "") as str(x)

            // Python owns the format language. To send the value there costs one call, and gives every
            // spec with Python's own rounding and its own error messages.
            using Py.GILState gil = PythonRuntime.GIL();
            using PyObject text_py = Call(Module, "format", [value!, format_spec]);
            return text_py.As<string>();
        }
    }

    /// <summary>
    ///     The C# form of a Python f-string. The compiler turns a <c>$"..."</c> that is passed to
    ///     <see cref="builtins.print(fstring)"/> into calls of this type: one for each piece of text and one
    ///     for each hole. A hole is written by <see cref="builtins.str"/>, or by <see cref="builtins.format"/>
    ///     when it has a format spec, so the specs are Python's and not .NET's.
    /// </summary>
    /// <remarks>
    ///     C#'s alignment syntax (<c>{x,10}</c>) does not compile here: Python writes it inside the spec
    ///     (<c>{x:&gt;10}</c>).
    /// </remarks>
    [InterpolatedStringHandler]
    internal readonly struct fstring
    {
        /// <summary>The text so far.</summary>
        private readonly StringBuilder _text;

        /// <summary>Starts an f-string. The compiler calls it.</summary>
        /// <param name="literalLength">The number of characters outside the holes.</param>
        /// <param name="formattedCount">The number of holes.</param>
        public fstring(int literalLength, int formattedCount) => _text = new StringBuilder(literalLength + 16 * formattedCount);

        /// <summary>Adds a piece of text that stands between two holes.</summary>
        /// <param name="text">The text, as written.</param>
        public void AppendLiteral(string text) => _text.Append(text);

        /// <summary>Adds a hole without a format spec: <c>{value}</c>.</summary>
        /// <typeparam name="T">The type of the hole.</typeparam>
        /// <param name="value">The value of the hole.</param>
        /// <exception cref="NotSupportedException">The type has no Python form in this tree.</exception>
        public void AppendFormatted<T>(T value) => _text.Append(builtins.str(value));

        /// <summary>Adds a hole with a format spec: <c>{value:.4f}</c>.</summary>
        /// <typeparam name="T">The type of the hole.</typeparam>
        /// <param name="value">The value of the hole.</param>
        /// <param name="format">The spec after the colon.</param>
        /// <exception cref="PythonException">Python refuses the spec for this value.</exception>
        /// <exception cref="NotSupportedException">The type has no Python form in this tree.</exception>
        public void AppendFormatted<T>(T value, string? format) => _text.Append(builtins.format(value, format ?? ""));

        /// <summary>The finished text.</summary>
        /// <returns>The text, or an empty string for a value that the compiler did not build.</returns>
        public override string ToString() => _text?.ToString() ?? "";
    }
}
