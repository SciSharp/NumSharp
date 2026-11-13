using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     The answers the pinned NumPy (<c>numpy-version.txt</c>) and PyTorch gave for every array and tensor state the
    ///     state-matrix suites cross, read from
    ///     <c>Fixtures/torch_interop_states.json</c> (written by <c>test/oracle/gen_torch_interop_states.py</c>, committed, so
    ///     no Python runs at test time).
    /// </summary>
    /// <remarks>
    ///     Two catalogues, each keyed by state name and then by dtype name: <see cref="ArrayStates"/> (a NumPy array's own
    ///     layout and flags, and what <c>torch.from_numpy</c> made of it) and <see cref="TensorStates"/> (a tensor's own layout,
    ///     and what <c>t.numpy()</c> / <c>t.numpy(force=True)</c> made of it). The C# suites rebuild each state by name
    ///     (<see cref="StateCatalog"/>), so a name here without a recipe there - or the reverse - is a failure of its own
    ///     (<c>StateCatalogSyncTests</c>).
    /// </remarks>
    public static class StateOracle
    {
        /// <summary>The parsed fixture, loaded once (the file is read-only test data).</summary>
        private static readonly Lazy<JsonDocument> Document = new Lazy<JsonDocument>(Load);

        /// <summary>The NumPy version the fixture was recorded with.</summary>
        public static string NumPyVersion => Document.Value.RootElement.GetProperty("numpy").GetString();

        /// <summary>The PyTorch version the fixture was recorded with.</summary>
        public static string TorchVersion => Document.Value.RootElement.GetProperty("torch").GetString();

        /// <summary>The array states: name → dtype name → record.</summary>
        public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, ArrayRecord>> ArrayStates => ArrayCache.Value;

        /// <summary>The tensor states: name → dtype name → record.</summary>
        public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, TensorRecord>> TensorStates => TensorCache.Value;

        /// <summary>The array states, parsed once.</summary>
        private static readonly Lazy<IReadOnlyDictionary<string, IReadOnlyDictionary<string, ArrayRecord>>> ArrayCache =
            new Lazy<IReadOnlyDictionary<string, IReadOnlyDictionary<string, ArrayRecord>>>(() => Parse(
                "numpy_states", e => new ArrayRecord(Layout.Of(e.GetProperty("numpy")), FromNumpy.Of(e.GetProperty("from_numpy")))));

        /// <summary>The tensor states, parsed once.</summary>
        private static readonly Lazy<IReadOnlyDictionary<string, IReadOnlyDictionary<string, TensorRecord>>> TensorCache =
            new Lazy<IReadOnlyDictionary<string, IReadOnlyDictionary<string, TensorRecord>>>(() => Parse(
                "torch_states", e => new TensorRecord(
                    TensorLayout.Of(e.GetProperty("tensor")),
                    ToNumpy.Of(e.GetProperty("numpy")),
                    e.TryGetProperty("numpy_force", out JsonElement force) ? ToNumpy.Of(force) : null)));

        /// <summary>Reads the fixture beside the test assembly.</summary>
        /// <returns>The parsed document.</returns>
        /// <exception cref="FileNotFoundException">The fixture was not copied to the output directory.</exception>
        private static JsonDocument Load()
        {
            string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "torch_interop_states.json");
            return JsonDocument.Parse(File.ReadAllText(path));
        }

        /// <summary>Parses one catalogue of the fixture.</summary>
        /// <typeparam name="T">The record type.</typeparam>
        /// <param name="section">The catalogue's property name.</param>
        /// <param name="read">Builds a record from one dtype row.</param>
        /// <returns>name → dtype → record.</returns>
        private static IReadOnlyDictionary<string, IReadOnlyDictionary<string, T>> Parse<T>(string section, Func<JsonElement, T> read)
        {
            var result = new Dictionary<string, IReadOnlyDictionary<string, T>>(StringComparer.Ordinal);
            foreach (JsonProperty state in Document.Value.RootElement.GetProperty(section).EnumerateObject())
            {
                var rows = new Dictionary<string, T>(StringComparer.Ordinal);
                foreach (JsonProperty row in state.Value.EnumerateObject())
                    rows[row.Name] = read(row.Value);
                result[state.Name] = rows;
            }
            return result;
        }

        /// <summary>A JSON array of integers.</summary>
        /// <param name="e">The element.</param>
        /// <returns>The values.</returns>
        private static long[] Longs(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt64()).ToArray();

        /// <summary>The five NumPy flags of an array.</summary>
        /// <param name="C">C_CONTIGUOUS.</param>
        /// <param name="F">F_CONTIGUOUS.</param>
        /// <param name="O">OWNDATA.</param>
        /// <param name="W">WRITEABLE.</param>
        /// <param name="A">ALIGNED.</param>
        public sealed record Flags(bool C, bool F, bool O, bool W, bool A)
        {
            /// <summary>Reads the flags object.</summary>
            /// <param name="e">The element.</param>
            /// <returns>The flags.</returns>
            public static Flags Of(JsonElement e) => new Flags(e.GetProperty("C").GetBoolean(), e.GetProperty("F").GetBoolean(),
                e.GetProperty("O").GetBoolean(), e.GetProperty("W").GetBoolean(), e.GetProperty("A").GetBoolean());

            /// <summary>The flags of a NumSharp array, read through <c>nd.flags</c>.</summary>
            /// <param name="nd">The array.</param>
            /// <returns>Its flags.</returns>
            public static Flags Of(NDArray nd)
            {
                var f = nd.flags;
                return new Flags(f.c_contiguous, f.f_contiguous, f.owndata, f.writeable, f.aligned);
            }

            /// <summary>The flags as NumPy prints them, for messages.</summary>
            /// <returns>"C F O W A" with each letter or a dash.</returns>
            public override string ToString() => $"{(C ? "C" : "-")}{(F ? "F" : "-")}{(O ? "O" : "-")}{(W ? "W" : "-")}{(A ? "A" : "-")}";
        }

        /// <summary>A NumPy array's shape, element strides and flags.</summary>
        /// <param name="Shape">The dimensions.</param>
        /// <param name="Strides">The strides in elements.</param>
        /// <param name="Flags">The five flags.</param>
        public sealed record Layout(long[] Shape, long[] Strides, Flags Flags)
        {
            /// <summary>Reads a layout record.</summary>
            /// <param name="e">The element.</param>
            /// <returns>The layout.</returns>
            public static Layout Of(JsonElement e) => new Layout(Longs(e.GetProperty("shape")), Longs(e.GetProperty("strides")), Flags.Of(e.GetProperty("flags")));
        }

        /// <summary>What <c>torch.from_numpy</c> did: a tensor's layout and whether it shared the array, or the error.</summary>
        /// <param name="Ok">Whether a tensor was made.</param>
        /// <param name="Error">The exception type's name when refused.</param>
        /// <param name="Message">The first line of the refusal.</param>
        /// <param name="Strides">The tensor's element strides.</param>
        /// <param name="IsContiguous">The tensor's <c>is_contiguous()</c>.</param>
        /// <param name="Shares">Whether the tensor's data pointer was the array's.</param>
        public sealed record FromNumpy(bool Ok, string Error, string Message, long[] Strides, bool IsContiguous, bool Shares)
        {
            /// <summary>Reads a from_numpy record.</summary>
            /// <param name="e">The element.</param>
            /// <returns>The record.</returns>
            public static FromNumpy Of(JsonElement e) => e.GetProperty("ok").GetBoolean()
                ? new FromNumpy(true, null, null, Longs(e.GetProperty("strides")), e.GetProperty("is_contiguous").GetBoolean(), e.GetProperty("shares").GetBoolean())
                : new FromNumpy(false, e.GetProperty("error").GetString(), e.GetProperty("message").GetString(), null, false, false);
        }

        /// <summary>One array state for one dtype.</summary>
        /// <param name="NumPy">The NumPy array's layout and flags.</param>
        /// <param name="FromNumpy">What torch.from_numpy made of it.</param>
        public sealed record ArrayRecord(Layout NumPy, FromNumpy FromNumpy);

        /// <summary>A tensor's own layout.</summary>
        /// <param name="Dtype">torch's dtype name.</param>
        /// <param name="Shape">The dimensions.</param>
        /// <param name="Strides">The element strides.</param>
        /// <param name="StorageOffset">The storage offset in elements.</param>
        /// <param name="IsContiguous">is_contiguous().</param>
        /// <param name="RequiresGrad">requires_grad.</param>
        public sealed record TensorLayout(string Dtype, long[] Shape, long[] Strides, long StorageOffset, bool IsContiguous, bool RequiresGrad)
        {
            /// <summary>Reads a tensor record.</summary>
            /// <param name="e">The element.</param>
            /// <returns>The layout.</returns>
            public static TensorLayout Of(JsonElement e) => new TensorLayout(e.GetProperty("dtype").GetString(), Longs(e.GetProperty("shape")),
                Longs(e.GetProperty("strides")), e.GetProperty("storage_offset").GetInt64(), e.GetProperty("is_contiguous").GetBoolean(),
                e.GetProperty("requires_grad").GetBoolean());
        }

        /// <summary>What <c>t.numpy()</c> returned (the array's layout, dtype and sharing) or raised.</summary>
        /// <param name="Ok">Whether an array was returned.</param>
        /// <param name="Error">The exception type's name when refused.</param>
        /// <param name="Message">The first line of the refusal.</param>
        /// <param name="Dtype">The array's NumPy dtype name.</param>
        /// <param name="Layout">The array's layout and flags.</param>
        /// <param name="Shares">Whether the array's data pointer was the tensor's.</param>
        public sealed record ToNumpy(bool Ok, string Error, string Message, string Dtype, Layout Layout, bool Shares)
        {
            /// <summary>Reads a numpy() record.</summary>
            /// <param name="e">The element.</param>
            /// <returns>The record.</returns>
            public static ToNumpy Of(JsonElement e) => e.GetProperty("ok").GetBoolean()
                ? new ToNumpy(true, null, null, e.GetProperty("dtype").GetString(), Layout.Of(e), e.GetProperty("shares").GetBoolean())
                : new ToNumpy(false, e.GetProperty("error").GetString(), e.GetProperty("message").GetString(), null, null, false);
        }

        /// <summary>One tensor state for one dtype.</summary>
        /// <param name="Tensor">The tensor's own layout.</param>
        /// <param name="NumPy">What t.numpy() did.</param>
        /// <param name="NumPyForce">What t.numpy(force=True) did, recorded only when numpy() refused.</param>
        public sealed record TensorRecord(TensorLayout Tensor, ToNumpy NumPy, ToNumpy NumPyForce);
    }
}
