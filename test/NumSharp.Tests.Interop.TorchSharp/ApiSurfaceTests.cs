using System;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.TorchSharp;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     API approval: the package's public surface, read by reflection, must equal the approved list below line for line. A
    ///     member added, removed, renamed or re-typed fails here until the list is updated on purpose.
    /// </summary>
    [TestClass]
    public class ApiSurfaceTests : TorchTestBase
    {
        /// <summary>The package assembly.</summary>
        private static readonly Assembly Package = typeof(NDArrayTorchInterop).Assembly;

        /// <summary>The approved public surface, in the format <see cref="Surface"/> produces.</summary>
        private static readonly string[] Approved =
        {
            "static class NumSharp.Interop.TorchSharp.NDArrayTorchInterop",
            "  static Boolean MayShareMemory(this NDArray array, Tensor tensor)",
            "  static DType ToDType(ScalarType scalarType)",
            "  static Int32 LiveExports { get; }",
            "  static Int32 LiveImports { get; }",
            "  static NDArray AsNDArray(this Tensor tensor, TorchCodecMode mode)",
            "  static NDArray ToNDArray(this Tensor tensor)",
            "  static ScalarType ToScalarType(DType dtype)",
            "  static Tensor AsTensor(this NDArray array, TorchCodecMode mode, Nullable<ScalarType> dtype, String casting)",
            "  static Tensor ToTensor(this NDArray array, Nullable<ScalarType> dtype, String casting)",
            "enum NumSharp.Interop.TorchSharp.TorchCodecMode : IComparable, IConvertible, IFormattable, ISpanFormattable",
        };

        /// <summary>The public surface, as reflection sees it, equals the approved list exactly.</summary>
        [TestMethod]
        public void PublicSurface_IsExactlyTheApprovedOne()
        {
            string[] actual = Surface(Package);
            actual.Should().Equal(Approved,
                "the public API changed; if that was intended, update Approved. Actual surface:\n" + string.Join("\n", actual));
        }

        /// <summary>The mode enum has exactly the three members of the other bridges' codec modes, with their values.</summary>
        [TestMethod]
        public void TorchCodecMode_HasTheThreeCodecModes()
        {
            Enum.GetNames(typeof(TorchCodecMode)).Should().Equal("Auto", "View", "Copy");
            ((int)TorchCodecMode.Auto).Should().Be(0, "Auto is the default(TorchCodecMode)");
            ((int)TorchCodecMode.View).Should().Be(1);
            ((int)TorchCodecMode.Copy).Should().Be(2);
        }

        /// <summary>
        ///     The assembly's public surface as sorted text: one line per exported type (kind, full name, interfaces), then one
        ///     indented line per public property and method it declares (accessors and enum fields excluded), sorted ordinally.
        /// </summary>
        /// <param name="assembly">The assembly to describe.</param>
        /// <returns>The lines.</returns>
        private static string[] Surface(Assembly assembly)
        {
            var lines = new System.Collections.Generic.List<string>();
            foreach (Type type in assembly.GetExportedTypes().OrderBy(t => t.FullName, StringComparer.Ordinal))
            {
                string kind = type.IsEnum ? "enum" : type.IsAbstract && type.IsSealed ? "static class" : type.IsSealed ? "sealed class" : type.IsInterface ? "interface" : "class";
                string interfaces = string.Join(", ", type.GetInterfaces().Select(Name).OrderBy(n => n, StringComparer.Ordinal));
                lines.Add($"{kind} {type.Namespace}.{Name(type)}{(interfaces.Length > 0 ? " : " + interfaces : "")}");
                if (type.IsEnum)
                    continue;

                const BindingFlags declared = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
                var members = type.GetProperties(declared)
                    .Select(p => $"  {(p.GetMethod?.IsStatic == true ? "static " : "")}{Name(p.PropertyType)} {p.Name} {{ {(p.GetMethod?.IsPublic == true ? "get; " : "")}{(p.SetMethod?.IsPublic == true ? "set; " : "")}}}")
                    .Concat(type.GetMethods(declared).Where(m => !m.IsSpecialName).Select(Signature))
                    .Concat(type.GetConstructors(BindingFlags.Public | BindingFlags.Instance).Select(c => $"  .ctor({Parameters(c)})"))
                    .OrderBy(s => s, StringComparer.Ordinal);
                lines.AddRange(members);
            }
            return lines.ToArray();
        }

        /// <summary>A method's line: static-ness, return type, name with generic parameters, and parameters.</summary>
        /// <param name="m">The method.</param>
        /// <returns>The line.</returns>
        private static string Signature(MethodInfo m)
        {
            string generics = m.IsGenericMethodDefinition ? "<" + string.Join(", ", m.GetGenericArguments().Select(a => a.Name)) + ">" : "";
            return $"  {(m.IsStatic ? "static " : "")}{Name(m.ReturnType)} {m.Name}{generics}({Parameters(m)})";
        }

        /// <summary>A parameter list, with <c>this</c> on an extension method's first parameter.</summary>
        /// <param name="method">The method or constructor.</param>
        /// <returns>The comma-separated parameters.</returns>
        private static string Parameters(MethodBase method)
        {
            bool extension = method.IsDefined(typeof(ExtensionAttribute));
            return string.Join(", ", method.GetParameters().Select((p, i) =>
                (i == 0 && extension ? "this " : "") + Name(p.ParameterType) + " " + p.Name));
        }

        /// <summary>A readable type name: generic arguments by name, arrays as <c>[]</c>.</summary>
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
