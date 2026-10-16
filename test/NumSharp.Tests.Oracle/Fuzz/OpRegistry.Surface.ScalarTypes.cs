using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     The surface tier's scalar-type section (<c>gen_surface_oracle.gen_scalar_types</c>): <c>np.issubdtype</c> and
    ///     <c>np.isdtype</c> over NumPy's scalar classes, <c>np.dtype</c> / <c>np.finfo</c> / <c>np.iinfo</c> of a class,
    ///     each class's own attributes and its isinstance / issubclass checks, and the relatives <c>np.typecodes</c>,
    ///     <c>np.sctypeDict</c>, <c>np.True_</c> / <c>np.False_</c> and <c>np.longdouble</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A case names a class or a dtype by a small JSON dict (<c>{"class": "integer"}</c>, <c>{"dtype": "i4"}</c>,
    ///     <c>null</c>). Each adapter calls the NumSharp overload a C# port of the NumPy call binds: a class is its
    ///     <see cref="NDScalarType"/> (<c>np.integer</c>), a dtype string stays a string — so the string overloads, and
    ///     the implicit string → <see cref="DType"/> conversion with its own refusal, are what run — and NumPy's
    ///     <c>None</c> is a null <see cref="DType"/>.
    ///     </para>
    ///     <para>
    ///     A list of classes is rendered as the Python repr of their names (<c>['integer', 'number', 'generic']</c>), the
    ///     form the generator records; the renderer only formats, it never repairs an answer.
    ///     </para>
    /// </remarks>
    public static partial class OpRegistry
    {
        /// <summary>The abstract class a generator name stands for (<c>"integer"</c> → <see cref="np.integer"/>).</summary>
        /// <param name="name">One of NumPy's ten abstract class names.</param>
        /// <returns>The class object.</returns>
        /// <exception cref="NotSupportedException">A name the generator does not emit.</exception>
        private static NDScalarType AbstractScalarClass(string name) => name switch
        {
            "generic" => np.generic,
            "number" => np.number,
            "integer" => np.integer,
            "signedinteger" => np.signedinteger,
            "unsignedinteger" => np.unsignedinteger,
            "inexact" => np.inexact,
            "floating" => np.floating,
            "complexfloating" => np.complexfloating,
            "flexible" => np.flexible,
            "character" => np.character,
            _ => throw new NotSupportedException($"scalar class '{name}' is not one of NumPy's abstract classes"),
        };

        /// <summary>
        ///     <c>np.issubdtype(arg1, arg2)</c> over two argument dicts, through the overload each pair of spellings binds:
        ///     class / class, class / dtype string, dtype string / class, dtype string / dtype string, and <c>None</c> as a
        ///     null <see cref="DType"/>.
        /// </summary>
        /// <param name="p">The case params: <c>arg1</c>, <c>arg2</c>.</param>
        /// <returns>NumSharp's answer as a 0-d bool array.</returns>
        /// <exception cref="NotSupportedException">A refused dtype string (NumSharp's dtype parser), the outcome the case may expect.</exception>
        internal static NDArray ScalarTypeIsSubdtype(IReadOnlyDictionary<string, JsonElement> p)
        {
            var a = p["arg1"];
            var b = p["arg2"];
            bool r;
            if (IsClassArg(a) && IsClassArg(b))
                r = np.issubdtype(AbstractScalarClass(a.GetProperty("class").GetString()), AbstractScalarClass(b.GetProperty("class").GetString()));
            else if (IsClassArg(a))
                r = np.issubdtype(AbstractScalarClass(a.GetProperty("class").GetString()), DtypeArg(b));
            else if (IsClassArg(b))
                r = np.issubdtype(DtypeArg(a), AbstractScalarClass(b.GetProperty("class").GetString()));
            else if (a.ValueKind == JsonValueKind.Null)
                r = np.issubdtype((DType)null, DtypeString(b));
            else
                r = np.issubdtype(DtypeString(a), DtypeString(b));
            return NDArray.Scalar(r);
        }

        /// <summary>
        ///     <c>np.isdtype(arg, kind)</c> / <c>np.isdtype(arg, (kind, ...))</c>: a single kind binds the overload its
        ///     spelling picks (a name, a class, a descriptor), several the mixed <c>params object[]</c> form.
        /// </summary>
        /// <param name="p">The case params: <c>arg</c> (a class or a dtype dict) and <c>kinds</c> (a list of kind dicts).</param>
        /// <returns>NumSharp's answer as a 0-d bool array.</returns>
        /// <exception cref="ValueError">An unknown kind name, the outcome the case may expect.</exception>
        /// <exception cref="TypeError">A kind of a refused type (an int), the outcome the case may expect.</exception>
        internal static NDArray ScalarTypeIsDtype(IReadOnlyDictionary<string, JsonElement> p)
        {
            var kinds = p["kinds"].EnumerateArray().Select(KindObject).ToArray();
            var arg = p["arg"];
            bool r;
            if (IsClassArg(arg))
            {
                var a = AbstractScalarClass(arg.GetProperty("class").GetString());
                r = kinds.Length != 1 ? np.isdtype(a, kinds) : kinds[0] switch
                {
                    string s => np.isdtype(a, s),
                    NDScalarType c => np.isdtype(a, c),
                    DType d => np.isdtype(a, d),
                    var o => np.isdtype(a, new[] { o }),
                };
            }
            else
            {
                var a = np.dtype(arg.GetProperty("dtype").GetString());
                r = kinds.Length != 1 ? np.isdtype(a, kinds) : kinds[0] switch
                {
                    string s => np.isdtype(a, s),
                    NDScalarType c => np.isdtype(a, c),
                    DType d => np.isdtype(a, d),
                    var o => np.isdtype(a, new[] { o }),
                };
            }

            return NDArray.Scalar(r);
        }

        /// <summary>
        ///     A class's isinstance / issubclass check (<c>check</c> = <c>instance</c> / <c>subclass</c>): isinstance of the
        ///     case's operand (a 0-d array that stands for a NumPy scalar, or a 1-D array), issubclass of a class or of a
        ///     dtype's class (a <see cref="DType"/> converts to its class).
        /// </summary>
        /// <param name="p">The case params: <c>cls</c>, <c>check</c>, and <c>sub</c> for a subclass check.</param>
        /// <param name="ops">The operand of an isinstance check.</param>
        /// <returns>NumSharp's answer as a 0-d bool array.</returns>
        /// <exception cref="NotSupportedException">A check the generator does not emit.</exception>
        internal static NDArray ScalarTypeCheck(IReadOnlyDictionary<string, JsonElement> p, NDArray[] ops)
        {
            var cls = AbstractScalarClass(p["cls"].GetString());
            switch (p["check"].GetString())
            {
                case "instance":
                    return NDArray.Scalar(cls.__instancecheck__(ops[0]));
                case "subclass":
                {
                    var sub = p["sub"];
                    return NDArray.Scalar(IsClassArg(sub)
                        ? cls.__subclasscheck__(AbstractScalarClass(sub.GetProperty("class").GetString()))
                        : cls.__subclasscheck__(np.dtype(sub.GetProperty("dtype").GetString())));
                }
                default:
                    throw new NotSupportedException($"scalar-class check '{p["check"].GetString()}'");
            }
        }

        /// <summary>
        ///     One attribute of a class as text: an abstract class by name, or a concrete one as the class of a dtype
        ///     (<c>of</c>: <c>(NDScalarType)np.dtype(of)</c>, NumPy's <c>np.dtype(of).type</c>).
        /// </summary>
        /// <param name="p">The case params: <c>cls</c>, <c>attr</c>, optionally <c>of</c>.</param>
        /// <returns>The attribute's text (a class list as the repr of its names).</returns>
        /// <exception cref="NotSupportedException">An attribute the generator does not emit.</exception>
        internal static string ScalarTypeAttribute(IReadOnlyDictionary<string, JsonElement> p)
        {
            NDScalarType cls = p.TryGetValue("of", out var of)
                ? np.dtype(of.GetString())
                : AbstractScalarClass(p["cls"].GetString());
            return p["attr"].GetString() switch
            {
                "__name__" => cls.__name__,
                "__qualname__" => cls.__qualname__,
                "__module__" => cls.__module__,
                "repr" => cls.ToString(),
                "__mro__" => ClassNames(cls.__mro__),
                "__bases__" => ClassNames(cls.__bases__),
                "__base__" => cls.__base__?.__name__ ?? "None",
                "mro" => ClassNames(cls.mro()),
                "__subclasses__" => ClassNames(cls.__subclasses__()),
                var a => throw new NotSupportedException($"scalar-class attribute '{a}'"),
            };
        }

        /// <summary><c>np.typecodes[key]</c>, or the repr of its key list for a null key.</summary>
        /// <param name="p">The case params: <c>key</c>.</param>
        /// <returns>The type-code string, or the key list's repr.</returns>
        /// <exception cref="KeyNotFoundException">A key NumSharp's table lacks.</exception>
        internal static string TypeCodes(IReadOnlyDictionary<string, JsonElement> p)
        {
            var key = p["key"];
            return key.ValueKind == JsonValueKind.Null
                ? "[" + string.Join(", ", np.typecodes.Keys.Select(k => "'" + k + "'")) + "]"
                : np.typecodes[key.GetString()];
        }

        /// <summary>
        ///     The dtype a class converts to, through <c>np.dtype</c> / <c>np.finfo</c> / <c>np.iinfo</c> (<c>fn</c>) — an
        ///     abstract class by name (<c>sctype</c>, refused) or a dtype's class (<c>sctype_of</c>).
        /// </summary>
        /// <param name="fn">"dtype", "finfo" or "iinfo".</param>
        /// <param name="p">The case params: <c>sctype</c> or <c>sctype_of</c>.</param>
        /// <returns>The resulting dtype's code.</returns>
        /// <exception cref="TypeError">np.dtype of an abstract class (the outcome the case expects).</exception>
        /// <exception cref="ValueError">np.finfo / np.iinfo of an abstract class (the outcome the case expects).</exception>
        /// <exception cref="NotSupportedException">A function the generator does not emit.</exception>
        internal static NPTypeCode ScalarTypeDtype(string fn, IReadOnlyDictionary<string, JsonElement> p)
        {
            NDScalarType cls = p.TryGetValue("sctype", out var name)
                ? AbstractScalarClass(name.GetString())
                : np.dtype(p["sctype_of"].GetString());
            return fn switch
            {
                "dtype" => np.dtype(cls).typecode,
                "finfo" => np.finfo(cls).dtype.typecode,
                "iinfo" => np.iinfo(cls).dtype.typecode,
                _ => throw new NotSupportedException($"scalar-class conversion '{fn}'"),
            };
        }

        /// <summary>
        ///     One facet of NumPy's bool scalar <c>np.True_</c> / <c>np.False_</c> (a C# <see cref="bool"/> in NumSharp): the
        ///     value, its inversion (NumPy's <c>~np.True_</c>, C#'s <c>!</c>), its integer value (<c>int(np.True_)</c>) and
        ///     the array it converts to (<c>np.asarray(np.True_)</c>, a 0-d bool array).
        /// </summary>
        /// <param name="value">The constant under test (<see cref="np.True_"/> or <see cref="np.False_"/>).</param>
        /// <param name="p">The case params: <c>facet</c>.</param>
        /// <returns>The facet as an array (a scalar as a 0-d array).</returns>
        /// <exception cref="NotSupportedException">A facet the generator does not emit.</exception>
        internal static NDArray BoolScalarFacet(bool value, IReadOnlyDictionary<string, JsonElement> p) => p["facet"].GetString() switch
        {
            "value" => NDArray.Scalar(value),
            "invert" => NDArray.Scalar(!value),
            "int" => NDArray.Scalar(value ? 1L : 0L),
            "asarray" => np.asarray(value),
            var f => throw new NotSupportedException($"bool scalar facet '{f}'"),
        };

        /// <summary>
        ///     An array-valued facet of <c>np.longdouble</c>: its item size, whether it is a floating class, and an array
        ///     built with it as the dtype (<c>np.zeros(3, np.longdouble)</c>). The dtype facet itself is a dtype-kind case
        ///     (<see cref="ApplySurfaceDtype"/>).
        /// </summary>
        /// <param name="p">The case params: <c>facet</c>.</param>
        /// <returns>The facet as an array (a scalar as a 0-d array).</returns>
        /// <exception cref="NotSupportedException">A facet the generator does not emit.</exception>
        internal static NDArray LongDoubleFacet(IReadOnlyDictionary<string, JsonElement> p) => p["facet"].GetString() switch
        {
            "itemsize" => NDArray.Scalar((long)np.longdouble.itemsize),
            "is_floating" => NDArray.Scalar(np.issubdtype(np.longdouble, np.floating)),
            "zeros" => np.zeros(3, np.longdouble),
            var f => throw new NotSupportedException($"np.longdouble facet '{f}'"),
        };

        /// <summary>True when a case names its dtype through a scalar class (the <c>sctype</c> / <c>sctype_of</c> params).</summary>
        /// <param name="p">The case params.</param>
        /// <returns>Whether the scalar-type adapters own the case.</returns>
        internal static bool IsScalarTypeDtypeCase(IReadOnlyDictionary<string, JsonElement> p)
            => p.ContainsKey("sctype") || p.ContainsKey("sctype_of");

        /// <summary>True for an argument dict that names an abstract class (<c>{"class": ...}</c>).</summary>
        /// <param name="e">The argument (an object or null).</param>
        /// <returns>Whether it is a class dict.</returns>
        private static bool IsClassArg(JsonElement e) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty("class", out _);

        /// <summary>
        ///     A dtype argument as the <see cref="DType"/> a C# port passes: the string converted IMPLICITLY at the call (so
        ///     NumSharp's own refusal of a bad string runs, as NumPy's does inside the call), or null for <c>None</c>.
        /// </summary>
        /// <param name="e">The argument dict or null.</param>
        /// <returns>The descriptor, or null.</returns>
        /// <exception cref="NotSupportedException">The dtype string is refused (the outcome a case may expect).</exception>
        private static DType DtypeArg(JsonElement e) => e.ValueKind == JsonValueKind.Null ? null : (DType)e.GetProperty("dtype").GetString();

        /// <summary>A dtype argument's string, or null for <c>None</c>.</summary>
        /// <param name="e">The argument dict or null.</param>
        /// <returns>The dtype string, or null.</returns>
        private static string DtypeString(JsonElement e) => e.ValueKind == JsonValueKind.Null ? null : e.GetProperty("dtype").GetString();

        /// <summary>An isdtype kind dict as the object a C# port passes: a name, a class, a descriptor, or (refused) an int.</summary>
        /// <param name="e">The kind dict.</param>
        /// <returns>The kind object.</returns>
        /// <exception cref="NotSupportedException">A kind dict the generator does not emit.</exception>
        private static object KindObject(JsonElement e)
        {
            if (e.TryGetProperty("name", out var name)) return name.GetString();
            if (e.TryGetProperty("int", out var i)) return i.GetInt32();
            if (e.TryGetProperty("class", out var c)) return AbstractScalarClass(c.GetString());
            if (e.TryGetProperty("dtype", out var d)) return np.dtype(d.GetString());
            throw new NotSupportedException($"isdtype kind {e}");
        }

        /// <summary>The Python repr of a list of class names: <c>['integer', 'number', 'generic']</c>.</summary>
        /// <param name="classes">The classes.</param>
        /// <returns>The repr.</returns>
        private static string ClassNames(IEnumerable<NDScalarType> classes)
            => "[" + string.Join(", ", classes.Select(c => "'" + c.__name__ + "'")) + "]";
    }
}
