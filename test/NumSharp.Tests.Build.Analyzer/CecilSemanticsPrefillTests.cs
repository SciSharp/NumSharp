using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Mono.Cecil;
using NumSharp.Build;

namespace NumSharp.Tests.Build.Analyzer
{
    /// <summary>
    ///     <see cref="CecilSemanticsPrefill"/> stores, in linear time, the property and event accessors Cecil would read in
    ///     quadratic time - and must store exactly what Cecil's own reader produces. Each test reads one assembly twice
    ///     (Cecil's lazy path, and the prefill for every type), compares every accessor and every method's semantics, and
    ///     writes both modules and compares the bytes: the prefill may change only how fast the answers arrive.
    /// </summary>
    [TestClass]
    public class CecilSemanticsPrefillTests
    {
        /// <summary>A fixture with every accessor shape C# emits, in types large enough to be prefilled by default.</summary>
        private static string AccessorZoo()
        {
            var sb = new StringBuilder("using System;\nnamespace Zoo {\n");
            sb.Append("public interface IHas { int P { get; set; } event EventHandler E; int this[int i] { get; } }\n");
            sb.Append("public class Big : IHas {\n");
            for (int i = 0; i < 40; i++)
                sb.Append($"  public int Auto{i} {{ get; set; }}\n  public int GetOnly{i} => {i};\n  private int _s{i}; public int SetOnly{i} {{ set => _s{i} = value; }}\n");
            for (int i = 0; i < 10; i++)
                sb.Append($"  public int Init{i} {{ get; init; }}\n  public static int Static{i} {{ get; set; }}\n");
            for (int i = 0; i < 20; i++)
                sb.Append($"  public event EventHandler Field{i};\n  public event EventHandler Custom{i} {{ add {{ }} remove {{ }} }}\n");
            sb.Append("  public int this[string k] => k.Length;\n  public int this[int a, int b] { get => a + b; set { } }\n");
            sb.Append("  int IHas.P { get; set; }\n  event EventHandler IHas.E { add { } remove { } }\n  int IHas.this[int i] => i;\n");
            sb.Append("  public void Plain() { }\n  public class Nested {\n");
            for (int i = 0; i < 70; i++)
                sb.Append($"    public string N{i} {{ get; set; }}\n");
            sb.Append("  }\n}\npublic class Generic<T> {\n");
            for (int i = 0; i < 70; i++)
                sb.Append($"  public T G{i} {{ get; set; }}\n");
            sb.Append("}\npublic class Small { public int A { get; set; } public event EventHandler B; }\n}\n");
            return sb.ToString();
        }

        /// <summary>Reads an assembly the way the weaver does (in memory, deferred, with symbols when present).</summary>
        /// <param name="path">The assembly.</param>
        /// <returns>The assembly definition (the caller disposes it).</returns>
        private static AssemblyDefinition Read(string path)
            => AssemblyDefinition.ReadAssembly(path, new ReaderParameters
            {
                InMemory = true,
                ReadSymbols = File.Exists(Path.ChangeExtension(path, ".pdb")),
                ReadingMode = ReadingMode.Deferred,
            });

        /// <summary>Every type of a module, nested types included.</summary>
        /// <param name="module">The module.</param>
        /// <returns>The types in a stable order (outer first).</returns>
        private static IEnumerable<TypeDefinition> AllTypes(ModuleDefinition module)
        {
            foreach (var type in module.Types)
                foreach (var t in WithNested(type))
                    yield return t;
        }

        /// <summary>A type followed by its nested types, depth-first.</summary>
        /// <param name="type">The type.</param>
        /// <returns>The type and everything nested in it.</returns>
        private static IEnumerable<TypeDefinition> WithNested(TypeDefinition type)
        {
            yield return type;
            foreach (var nested in type.NestedTypes)
                foreach (var t in WithNested(nested))
                    yield return t;
        }

        /// <summary>A comparable rendering of everything the semantics decide for a module: each property's and event's accessors, and each method's semantics.</summary>
        /// <param name="module">The module.</param>
        /// <returns>One line per member.</returns>
        private static List<string> Render(ModuleDefinition module)
        {
            var lines = new List<string>();
            foreach (var type in AllTypes(module))
            {
                foreach (var p in type.Properties)
                    lines.Add($"P {p.FullName} get={p.GetMethod?.MetadataToken} set={p.SetMethod?.MetadataToken} other={p.OtherMethods.Count}");
                foreach (var e in type.Events)
                    lines.Add($"E {e.FullName} add={e.AddMethod?.MetadataToken} remove={e.RemoveMethod?.MetadataToken} raise={e.InvokeMethod?.MetadataToken} other={e.OtherMethods.Count}");
                foreach (var m in type.Methods)
                    lines.Add($"M {m.FullName} {m.SemanticsAttributes}");
            }

            return lines;
        }

        /// <summary>Writes a module to memory (without symbols: the comparison is of the metadata and IL the semantics shape).</summary>
        /// <param name="assembly">The assembly.</param>
        /// <returns>The written bytes.</returns>
        private static byte[] WriteBytes(AssemblyDefinition assembly)
        {
            using var ms = new MemoryStream();
            assembly.Write(ms, new WriterParameters { WriteSymbols = false, DeterministicMvid = true });
            return ms.ToArray();
        }

        /// <summary>
        ///     Every accessor shape C# emits (auto, get-only, set-only, init, static, field-like and custom events,
        ///     indexer overloads, explicit interface implementations, nested and generic types): prefilled, Cecil's view
        ///     and the written bytes are identical to its own reader's.
        /// </summary>
        [TestMethod]
        public void Prefill_EveryAccessorShape_MatchesCecilsOwnReader()
        {
            var fixture = WeaverTestHarness.Compile(AccessorZoo(), "PrefillZoo");

            using var lazy = Read(fixture.DllPath);
            var expected = Render(lazy.MainModule);
            var expectedBytes = WriteBytes(lazy);

            using var prefilled = Read(fixture.DllPath);
            int count = CecilSemanticsPrefill.Prefill(prefilled.MainModule, fixture.DllPath, threshold: 0);
            Assert.IsTrue(count >= 5, $"the zoo's types must be prefilled (prefilled {count})");
            CollectionAssert.AreEqual(expected, Render(prefilled.MainModule));
            CollectionAssert.AreEqual(expectedBytes, WriteBytes(prefilled), "the written assembly must not change");
        }

        /// <summary>The default threshold prefills the large types only: <c>Small</c> (two members) keeps Cecil's own path, and the answers still match.</summary>
        [TestMethod]
        public void Prefill_DefaultThreshold_SkipsSmallTypes()
        {
            var fixture = WeaverTestHarness.Compile(AccessorZoo(), "PrefillThreshold");
            using var lazy = Read(fixture.DllPath);
            var expected = Render(lazy.MainModule);

            using var prefilled = Read(fixture.DllPath);
            int count = CecilSemanticsPrefill.Prefill(prefilled.MainModule, fixture.DllPath);
            Assert.AreEqual(3, count, "Big, Big.Nested and Generic<T> reach the threshold; Small and IHas do not");
            CollectionAssert.AreEqual(expected, Render(prefilled.MainModule));
        }

        /// <summary>A type whose semantics Cecil already read is left alone (its state is final), and the module's answers are unchanged.</summary>
        [TestMethod]
        public void Prefill_TypeAlreadyRead_IsLeftToCecil()
        {
            var fixture = WeaverTestHarness.Compile(AccessorZoo(), "PrefillAlreadyRead");
            using var lazy = Read(fixture.DllPath);
            var expected = Render(lazy.MainModule);

            using var prefilled = Read(fixture.DllPath);
            var big = prefilled.MainModule.GetType("Zoo.Big");
            _ = big.Properties[0].GetMethod; // Cecil reads Big's semantics now
            int count = CecilSemanticsPrefill.Prefill(prefilled.MainModule, fixture.DllPath, threshold: 0);
            int all = AllTypes(prefilled.MainModule).Count(t => t.HasProperties || t.HasEvents);
            Assert.AreEqual(all - 1, count, "every type with members but Big is prefilled");
            CollectionAssert.AreEqual(expected, Render(prefilled.MainModule));
        }

        /// <summary>
        ///     NumSharp itself - thousands of properties and events of real compiler output - prefilled in every type, gives
        ///     Cecil's answers for every member.
        /// </summary>
        [TestMethod]
        public void Prefill_NumSharpAssembly_MatchesCecilsOwnReader()
        {
            var path = typeof(NumSharp.NDArray).Assembly.Location;
            using var lazy = Read(path);
            var expected = Render(lazy.MainModule);

            using var prefilled = Read(path);
            int count = CecilSemanticsPrefill.Prefill(prefilled.MainModule, path, threshold: 0);
            Assert.IsTrue(count > 100, $"NumSharp has hundreds of types with properties (prefilled {count})");
            var actual = Render(prefilled.MainModule);
            Assert.AreEqual(expected.Count, actual.Count);
            for (int i = 0; i < expected.Count; i++)
                Assert.AreEqual(expected[i], actual[i]);
        }

        /// <summary>A module whose file cannot be read a second time is reported to the caller (the weaver then keeps Cecil's path): the prefill never half-applies.</summary>
        [TestMethod]
        public void Prefill_MissingFile_ThrowsBeforeTouchingTheModule()
        {
            var fixture = WeaverTestHarness.Compile(AccessorZoo(), "PrefillMissing");
            using var lazy = Read(fixture.DllPath);
            var expected = Render(lazy.MainModule);

            using var module = Read(fixture.DllPath);
            Assert.ThrowsException<FileNotFoundException>(() =>
                CecilSemanticsPrefill.Prefill(module.MainModule, Path.Combine(fixture.Directory, "absent.dll"), threshold: 0));
            CollectionAssert.AreEqual(expected, Render(module.MainModule));
        }
    }
}
