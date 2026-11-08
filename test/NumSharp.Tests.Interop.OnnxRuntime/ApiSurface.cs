using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Renders an assembly's public surface as sorted text for API-approval tests: one line per exported type (kind,
    ///     full name, base type and interfaces), then one indented line per public field, property, method and
    ///     constructor it declares — with static-ness, <c>this</c> on extension methods, <c>params</c>, by-reference
    ///     modifiers and DEFAULT VALUES, so a changed default (an API break for every caller that relied on it) fails the
    ///     approval as surely as a renamed member does. Shared by the interop test projects (this file is linked into the
    ///     ML.NET suite), which is why it lives in the common <c>NumSharp.Tests</c> namespace.
    /// </summary>
    internal static class ApiSurface
    {
        /// <summary>The assembly's public surface, one line per type and member, types and members sorted ordinally.</summary>
        /// <param name="assembly">The assembly to describe.</param>
        /// <returns>The lines, ready to compare with an approved list.</returns>
        internal static string[] Describe(Assembly assembly)
        {
            var lines = new List<string>();
            foreach (Type type in assembly.GetExportedTypes().OrderBy(t => t.FullName, StringComparer.Ordinal))
            {
                lines.Add(TypeLine(type));
                const BindingFlags declared = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
                // An enum's instance field is its value__ backing field, not API; its members are the static literals.
                IEnumerable<string> members = type.GetFields(declared).Where(f => !type.IsEnum || f.IsStatic).Select(FieldLine)
                    .Concat(type.GetProperties(declared).Select(PropertyLine))
                    // Accessors and operators are special names; properties are listed above, and no operator is public here.
                    .Concat(type.GetMethods(declared).Where(m => !m.IsSpecialName).Select(MethodLine))
                    .Concat(type.GetConstructors(BindingFlags.Public | BindingFlags.Instance).Select(c => $"  .ctor({Parameters(c)})"))
                    .OrderBy(s => s, StringComparer.Ordinal);
                lines.AddRange(members);
            }
            return lines.ToArray();
        }

        /// <summary>A type's line: its kind, namespace-qualified name, base class and the interfaces it adds.</summary>
        /// <param name="type">The exported type.</param>
        /// <returns>The line.</returns>
        private static string TypeLine(Type type)
        {
            string kind = type.IsEnum ? "enum"
                : type.IsValueType ? "struct"
                : type.IsInterface ? "interface"
                : type.IsAbstract && type.IsSealed ? "static class"
                : type.IsSealed ? "sealed class"
                : type.IsAbstract ? "abstract class"
                : "class";
            var bases = new List<string>();
            if (type.IsClass && type.BaseType is not null && type.BaseType != typeof(object))
                bases.Add(Name(type.BaseType));
            // Only the interfaces this type adds: those its base already implements are part of the base's line.
            bases.AddRange(type.GetInterfaces()
                .Where(i => type.BaseType is null || !i.IsAssignableFrom(type.BaseType))
                .Select(Name)
                .OrderBy(n => n, StringComparer.Ordinal));
            return $"{kind} {type.Namespace}.{Name(type)}{(bases.Count > 0 ? " : " + string.Join(", ", bases) : "")}";
        }

        /// <summary>A field's line, with <c>const</c> fields' values.</summary>
        /// <param name="f">The field.</param>
        /// <returns>The line.</returns>
        private static string FieldLine(FieldInfo f)
        {
            string modifiers = (f.IsLiteral ? "const " : f.IsStatic ? "static " : "") + (f.IsInitOnly ? "readonly " : "");
            return $"  {modifiers}{Name(f.FieldType)} {f.Name}{(f.IsLiteral ? " = " + Literal(f.GetRawConstantValue()) : "")}";
        }

        /// <summary>A property's (or indexer's) line, listing only its PUBLIC accessors.</summary>
        /// <param name="p">The property.</param>
        /// <returns>The line.</returns>
        private static string PropertyLine(PropertyInfo p)
        {
            ParameterInfo[] index = p.GetIndexParameters();
            string name = index.Length > 0 ? "this[" + string.Join(", ", index.Select(i => Name(i.ParameterType) + " " + i.Name)) + "]" : p.Name;
            string accessors = (p.GetMethod?.IsPublic == true ? "get; " : "") + (p.SetMethod?.IsPublic == true ? "set; " : "");
            return $"  {(p.GetMethod?.IsStatic == true ? "static " : "")}{Name(p.PropertyType)} {name} {{ {accessors}}}";
        }

        /// <summary>A method's line: static / abstract / override / virtual, return type, name with generic parameters, and parameters.</summary>
        /// <param name="m">The method.</param>
        /// <returns>The line.</returns>
        private static string MethodLine(MethodInfo m)
        {
            string generics = m.IsGenericMethodDefinition ? "<" + string.Join(", ", m.GetGenericArguments().Select(a => a.Name)) + ">" : "";
            string modifiers = m.IsStatic ? "static "
                : m.IsAbstract ? "abstract "
                : m.GetBaseDefinition() != m ? "override "
                : m.IsVirtual && !m.IsFinal ? "virtual "
                : "";
            return $"  {modifiers}{Name(m.ReturnType)} {m.Name}{generics}({Parameters(m)})";
        }

        /// <summary>
        ///     A parameter list: <c>this</c> on an extension method's first parameter, <c>params</c>, <c>in</c> / <c>out</c> /
        ///     <c>ref</c> on by-reference parameters, and each optional parameter's default value.
        /// </summary>
        /// <param name="method">The method or constructor.</param>
        /// <returns>The comma-separated parameters.</returns>
        private static string Parameters(MethodBase method)
        {
            bool extension = method.IsDefined(typeof(ExtensionAttribute));
            return string.Join(", ", method.GetParameters().Select((p, i) =>
                (i == 0 && extension ? "this " : "") +
                (p.IsDefined(typeof(ParamArrayAttribute)) ? "params " : "") +
                (p.ParameterType.IsByRef ? (p.IsIn ? "in " : p.IsOut ? "out " : "ref ") : "") +
                Name(p.ParameterType.IsByRef ? p.ParameterType.GetElementType() : p.ParameterType) + " " + p.Name +
                (p.HasDefaultValue ? " = " + Literal(p.DefaultValue) : "")));
        }

        /// <summary>A constant or default value as C# would spell it (invariant culture, quoted strings, lowercase booleans).</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its text.</returns>
        private static string Literal(object value) => value switch
        {
            null => "null",
            string s => "\"" + s + "\"",
            bool b => b ? "true" : "false",
            _ => Convert.ToString(value, CultureInfo.InvariantCulture),
        };

        /// <summary>A readable type name: generic arguments by name (<c>DenseTensor&lt;T&gt;</c>), arrays as <c>[]</c>.</summary>
        /// <param name="t">The type.</param>
        /// <returns>The name.</returns>
        private static string Name(Type t)
        {
            if (t.IsArray)
                return Name(t.GetElementType()) + "[]";
            if (t.IsGenericParameter || !t.IsGenericType)
                return t.Name;
            string bare = t.Name.Substring(0, t.Name.IndexOf('`'));
            return $"{bare}<{string.Join(", ", t.GetGenericArguments().Select(Name))}>";
        }
    }
}
