using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Interop.Tensors;

namespace NumSharp.Tests.Interop.Tensors
{
    /// <summary>
    ///     API approval: the package's public surface, read by reflection, must equal the approved list below line for
    ///     line. A member added, removed, renamed or re-typed — including a widened parameter or a lost <c>this</c> —
    ///     fails here until the list is updated on purpose, so the shipped NuGet surface cannot change by accident.
    ///     The package re-marks its whole surface <c>[Experimental("NUMSHARP_TENSORS")]</c>; that is checked too.
    /// </summary>
    [TestClass]
    public class ApiSurfaceTests : TensorsTestBase
    {
        /// <summary>The package assembly.</summary>
        private static readonly Assembly Package = typeof(NDArrayTensorsInterop).Assembly;

        /// <summary>The approved public surface, in the format <see cref="Surface"/> produces.</summary>
        private static readonly string[] Approved =
        {
            "static class NumSharp.Interop.Tensors.NDArrayTensorsInterop",
            "  static Int32 LiveExports { get; }",
            "  static Int32 LiveImports { get; }",
            "  static NDArray AsNDArray<T>(this Tensor<T> tensor)",
            "  static NDArray ToNDArray<T>(this ReadOnlyTensorSpan<T> span)",
            "  static NDArray ToNDArray<T>(this Tensor<T> tensor)",
            "  static NDArray ToNDArray<T>(this TensorSpan<T> span)",
            "  static NPTypeCode TypeCodeOf<T>()",
            "  static Tensor<T> ToTensor<T>(this NDArray source)",
            "  static TensorSpanHandle<T> AsTensorSpan<T>(this NDArray source)",
            "  static Type ToTensorElementClrType(DType dtype)",
            "sealed class NumSharp.Interop.Tensors.TensorSpanHandle<T> : IDisposable",
            "  Boolean IsDisposed { get; }",
            "  Boolean IsWriteable { get; }",
            "  IntPtr[] Lengths { get; }",
            "  IntPtr[] Strides { get; }",
            "  NDArray Source { get; }",
            "  ReadOnlyTensorSpan<T> ReadOnlySpan { get; }",
            "  TensorSpan<T> Span { get; }",
            "  Void Dispose()",
        };

        /// <summary>The public surface, as reflection sees it, equals the approved list exactly.</summary>
        [TestMethod]
        public void PublicSurface_IsExactlyTheApprovedOne()
        {
            string[] actual = Surface(Package);
            actual.Should().Equal(Approved,
                "the public API changed; if that was intended, update Approved. Actual surface:\n" + string.Join("\n", actual));
        }

        /// <summary>
        ///     Every public type carries <c>[Experimental("NUMSHARP_TENSORS")]</c> — which also covers its members — and
        ///     every extension verb carries it explicitly as well, so a consumer acknowledges the BCL's experimental
        ///     tensor types once, at this seam.
        /// </summary>
        [TestMethod]
        public void EveryPublicTypeAndVerb_IsMarkedExperimental()
        {
            foreach (Type type in Package.GetExportedTypes())
                ExperimentalId(type).Should().Be("NUMSHARP_TENSORS", $"{type.Name} must be [Experimental(\"NUMSHARP_TENSORS\")]");

            MethodInfo[] verbs = typeof(NDArrayTensorsInterop).GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Where(m => m.IsDefined(typeof(ExtensionAttribute))).ToArray();
            verbs.Should().HaveCount(6, "AsTensorSpan, ToTensor, ToNDArray x3, AsNDArray");
            foreach (MethodInfo verb in verbs)
                ExperimentalId(verb).Should().Be("NUMSHARP_TENSORS", $"{verb.Name}({verb.GetParameters()[0].ParameterType.Name}) must be [Experimental(\"NUMSHARP_TENSORS\")]");
        }

        /// <summary>The diagnostic id of a member's <see cref="ExperimentalAttribute"/>, or <c>null</c> when it has none.</summary>
        /// <param name="member">The type or method.</param>
        /// <returns>The diagnostic id.</returns>
        private static string ExperimentalId(MemberInfo member)
            => member.GetCustomAttribute<ExperimentalAttribute>()?.DiagnosticId;

        /// <summary>
        ///     The assembly's public surface as sorted text: one line per exported type (kind, full name, interfaces),
        ///     then one indented line per public property and method it declares (property accessors and the object
        ///     overrides excluded), sorted ordinally.
        /// </summary>
        /// <param name="assembly">The assembly to describe.</param>
        /// <returns>The lines.</returns>
        private static string[] Surface(Assembly assembly)
        {
            var lines = new System.Collections.Generic.List<string>();
            foreach (Type type in assembly.GetExportedTypes().OrderBy(t => t.FullName, StringComparer.Ordinal))
            {
                string kind = type.IsAbstract && type.IsSealed ? "static class" : type.IsSealed ? "sealed class" : type.IsInterface ? "interface" : "class";
                string interfaces = string.Join(", ", type.GetInterfaces().Select(Name).OrderBy(n => n, StringComparer.Ordinal));
                lines.Add($"{kind} {type.Namespace}.{Name(type)}{(interfaces.Length > 0 ? " : " + interfaces : "")}");

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

        /// <summary>A parameter list, with <c>this</c> on an extension method's first parameter and <c>in</c>/<c>ref</c> on by-reference ones.</summary>
        /// <param name="method">The method or constructor.</param>
        /// <returns>The comma-separated parameters.</returns>
        private static string Parameters(MethodBase method)
        {
            bool extension = method.IsDefined(typeof(ExtensionAttribute));
            return string.Join(", ", method.GetParameters().Select((p, i) =>
                (i == 0 && extension ? "this " : "") +
                (p.ParameterType.IsByRef ? (p.IsIn ? "in " : p.IsOut ? "out " : "ref ") : "") +
                Name(p.ParameterType.IsByRef ? p.ParameterType.GetElementType() : p.ParameterType) + " " + p.Name));
        }

        /// <summary>A readable type name: generic arguments by name (<c>Tensor&lt;T&gt;</c>), arrays as <c>[]</c>.</summary>
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
