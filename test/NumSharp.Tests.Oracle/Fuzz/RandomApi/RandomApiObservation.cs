using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace NumSharp.Tests.Fuzz.RandomApi
{
    /// <summary>
    ///     The canonical observation of a random-API result and of a receiver's state — the C# mirror of <c>obs</c> /
    ///     <c>bitgen_state_text</c> in <c>test/oracle/gen_random_oracle.py</c>. The corpus stores NumPy's observation; the
    ///     replay renders NumSharp's result the same way and compares the two trees.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Observation kinds: <c>none</c> (void / Python None), <c>array</c> (dtype, shape, C-order bytes as hex),
    ///     <c>float</c> (a Python float: the float64 bit pattern), <c>int</c> (a Python int: decimal text), <c>bool</c>,
    ///     <c>text</c>. NumPy answers a <c>size=None</c> draw with a Python scalar where NumSharp returns a 0-d array; the
    ///     observer renders a 0-d NumSharp result in the scalar kind the corpus expects, but only when its dtype is the one
    ///     that Python scalar implies (float64 for a float, an integer type for an int, bool for a bool) — anything else is
    ///     rendered as an array, so the comparison reports the dtype divergence instead of hiding it.
    ///     </para>
    ///     <para>
    ///     State text is canonical per engine (MT19937's 624-word key by SHA-256, the other engines in full) and, for a
    ///     RandomState, adds <c>has_gauss</c> and the cached Gaussian's bits. It is read through the public state
    ///     accessors, the same ones NumPy's side reads.
    ///     </para>
    /// </remarks>
    internal static class RandomApiObservation
    {
        /// <summary>
        ///     Renders a NumSharp result as an observation shaped like the expected one.
        /// </summary>
        /// <param name="value">The member's result (null for void).</param>
        /// <param name="returns">The member's canonical return type (from the inventory).</param>
        /// <param name="expected">NumPy's observation (only its kind is consulted, to pick the scalar rendering).</param>
        /// <returns>The observation.</returns>
        /// <exception cref="NotSupportedException">A result type the observer does not render yet.</exception>
        internal static JsonNode Observe(object value, string returns, JsonElement expected)
        {
            string want = expected.ValueKind == JsonValueKind.Object && expected.TryGetProperty("k", out var k) ? k.GetString() : null;
            if (returns == "void")
                return new JsonObject { ["k"] = "none" };
            // A C# scalar whose NumPy counterpart is a TYPED numpy scalar (the expected kind is "array": a generic member
            // such as randn<T>, recorded as NumPy's cast of the draw to T) is rendered as a 0-d array of its own dtype, so
            // the comparison sees the dtype as well as the value; otherwise it is the Python scalar kind.
            if (want == "array" && TypedScalar(value) is { } typed)
                return ArrayObsScalar(typed);
            // An `object`-typed result that is not a state object is SeedSequence.entropy: NumPy's int / list / array,
            // compared through its repr (NumSharp's own repr of the stored entropy).
            if (returns == "object" && value is not (null or NativeRandomState or NumPyRandom.State or BitGeneratorState))
                return new JsonObject { ["k"] = "text", ["v"] = ReprEntropy(value) };
            switch (value)
            {
                case null:
                    return new JsonObject { ["k"] = "none" };
                case NDArray nd:
                    return ObserveArray(nd, want);
                case string s:
                    return new JsonObject { ["k"] = "text", ["v"] = s };
                case bool b:
                    return new JsonObject { ["k"] = "bool", ["v"] = b };
                case double d:
                    return FloatObs(d);
                case float f:
                    return want == "float" ? FloatObs(f) : ArrayObsScalar(NDArray.Scalar(f));
                case int or long or uint or ulong or short or ushort or byte or sbyte or BigInteger or UInt128:
                    return new JsonObject { ["k"] = "int", ["v"] = Convert.ToString(value, CultureInfo.InvariantCulture) };
                case Half h:
                    return ArrayObsScalar(NDArray.Scalar(h));
                case Complex c:
                    return ArrayObsScalar(NDArray.Scalar(c));
                // Objects: by type and canonical state, the generator's obs() kinds.
                case BitGenerator bg:
                    return new JsonObject
                    {
                        ["k"] = "bitgen", ["type"] = bg.GetType().Name, ["state"] = BitGeneratorStateText(bg.state),
                        ["seed_seq"] = SeedSeqText(bg.seed_seq) ?? "None",
                    };
                case NumPyRandom rs:
                    return new JsonObject { ["k"] = "rs", ["str"] = rs.ToString(), ["state"] = RandomStateText((NumPyRandom.State)rs.get_state(false)) };
                case Generator g:
                    return new JsonObject
                    {
                        ["k"] = "gen", ["str"] = g.ToString(), ["state"] = BitGeneratorStateText(g.bit_generator.state),
                        ["seed_seq"] = SeedSeqText(g.bit_generator.seed_seq) ?? "None",
                    };
                case ISeedSequence seq:
                    return new JsonObject { ["k"] = "seedseq", ["v"] = SeedSeqText(seq) };
                // SeedSequence.spawn_key and .state: NumPy's tuple and dict, compared through their reprs.
                case BigInteger[] key:
                    return new JsonObject { ["k"] = "text", ["v"] = PyTupleRepr(key) };
                case Dictionary<string, object> dict:
                    return new JsonObject { ["k"] = "text", ["v"] = SeedSequenceStateRepr(dict) };
                case NativeRandomState t:
                    return new JsonObject { ["k"] = "legacy_state", ["v"] = LegacyTupleText(t) };
                case NumPyRandom.State st:
                    return new JsonObject { ["k"] = "rsdict", ["v"] = RsDictText(st) };
                case BitGeneratorState bgs:
                    return new JsonObject { ["k"] = "bgstate", ["v"] = BitGeneratorStateText(bgs) };
                // Managed integer arrays (the legacy tuple's key, seed words): NumPy's matching integer array.
                case uint[] a:
                    return ManagedArrayObs(a);
                case ulong[] a:
                    return ManagedArrayObs(a);
                case long[] a:
                    return ManagedArrayObs(a);
                case int[] a:
                    return ManagedArrayObs(a);
                // Object sequences (spawn's children): each item as its element type, against NumPy's list.
                case object[] items:
                {
                    string element = returns.EndsWith("[]", StringComparison.Ordinal) ? returns[..^2] : "object";
                    var list = new JsonArray();
                    for (int i = 0; i < items.Length; i++)
                    {
                        var itemExpected = expected.ValueKind == JsonValueKind.Object && expected.TryGetProperty("items", out var its)
                                           && i < its.GetArrayLength() ? its[i] : default;
                        list.Add(Observe(items[i], element, itemExpected));
                    }
                    return new JsonObject { ["k"] = "seq", ["items"] = list };
                }
            }
            throw new NotSupportedException($"the random-API observer does not render {value.GetType().Name} results yet");
        }

        /// <summary>
        ///     A seed sequence's canonical text — the generator's <c>seedseq_text</c>: the child counter, the mixed pool's
        ///     SHA-256 and NumSharp's repr (which reproduces NumPy's) as the last field; <c>SeedlessSeedSequence</c> for the
        ///     seedless one; null for none.
        /// </summary>
        /// <param name="seq">The sequence.</param>
        /// <returns>The text, or null.</returns>
        /// <exception cref="NotSupportedException">A custom <see cref="ISeedSequence"/> the observer cannot describe.</exception>
        internal static string SeedSeqText(ISeedSequence seq)
        {
            switch (seq)
            {
                case null:
                    return null;
                case SeedlessSeedSequence:
                    return "SeedlessSeedSequence";
                case SeedSequence ss:
                {
                    using var pool = ss.pool;
                    return $"SeedSequence|n_children_spawned={ss.n_children_spawned}|pool={Sha256Hex(pool.ToArray<uint>())}|repr={ss}";
                }
                default:
                    throw new NotSupportedException($"the random-API observer does not render {seq.GetType().Name} seed sequences");
            }
        }

        /// <summary>
        ///     The repr NumPy prints for a seed-sequence entropy value, through NumSharp's own repr of the stored entropy
        ///     (<c>SeedSequence.ReprEntropy</c>, the helper its <c>ToString</c> uses) — so the entropy getter is compared on
        ///     the same text the repr comparison already pins.
        /// </summary>
        /// <param name="entropy">The entropy object.</param>
        /// <returns>The repr text.</returns>
        private static string ReprEntropy(object entropy)
            => (string)ReprEntropyMethod.Invoke(null, new[] { entropy });

        /// <summary>SeedSequence's private static repr helper, resolved once.</summary>
        private static readonly System.Reflection.MethodInfo ReprEntropyMethod =
            typeof(SeedSequence).GetMethod("ReprEntropy", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
            ?? throw new InvalidOperationException("SeedSequence.ReprEntropy not found");

        /// <summary>Python's repr of a tuple of ints: <c>()</c>, <c>(5,)</c>, <c>(1, 2)</c>.</summary>
        /// <param name="items">The ints.</param>
        /// <returns>The repr.</returns>
        private static string PyTupleRepr(BigInteger[] items)
            => items.Length switch
            {
                0 => "()",
                1 => "(" + items[0].ToString(CultureInfo.InvariantCulture) + ",)",
                _ => "(" + string.Join(", ", items.Select(x => x.ToString(CultureInfo.InvariantCulture))) + ")",
            };

        /// <summary>
        ///     Python's repr of <c>SeedSequence.state</c>: <c>{'entropy': …, 'spawn_key': (…), 'pool_size': n,
        ///     'n_children_spawned': k}</c>, in NumPy's key order.
        /// </summary>
        /// <param name="d">NumSharp's state dictionary.</param>
        /// <returns>The repr.</returns>
        private static string SeedSequenceStateRepr(Dictionary<string, object> d)
            => "{'entropy': " + ReprEntropy(d["entropy"]) + ", 'spawn_key': " + PyTupleRepr((BigInteger[])d["spawn_key"]) +
               ", 'pool_size': " + Convert.ToString(d["pool_size"], CultureInfo.InvariantCulture) +
               ", 'n_children_spawned': " + Convert.ToString(d["n_children_spawned"], CultureInfo.InvariantCulture) + "}";

        /// <summary>A managed array rendered as the NumPy array of the same element type (1-D, C order).</summary>
        /// <param name="a">The array.</param>
        /// <returns>The array observation.</returns>
        private static JsonObject ManagedArrayObs(Array a)
        {
            using var nd = np.array(a);
            return ArrayObs(nd);
        }

        /// <summary>
        ///     The legacy state tuple's canonical text — the generator's <c>legacy_tuple_text</c>:
        ///     <c>tuple|algorithm|key=sha256|pos|has_gauss|gauss bits</c> (a null key reads <c>null</c>).
        /// </summary>
        /// <param name="t">The tuple.</param>
        /// <returns>The text.</returns>
        internal static string LegacyTupleText(NativeRandomState t)
            => $"tuple|{t.Algorithm}|key={(t.Key is null ? "null" : Sha256Hex(t.Key))}|pos={t.Pos}|has_gauss={t.HasGauss}|gauss={F64Bits(t.CachedGaussian)}";

        /// <summary>
        ///     RandomState's dict state as canonical text — the generator's <c>rsdict_text</c>: <c>rsdict|</c> plus the
        ///     RandomState state text, or <c>rsdict|null|…</c> when the bit generator state is unset (the parameterless
        ///     <see cref="NumPyRandom.State"/>).
        /// </summary>
        /// <param name="st">The dict state.</param>
        /// <returns>The text.</returns>
        internal static string RsDictText(NumPyRandom.State st)
            => st.state is null
                ? $"rsdict|null|has_gauss={st.has_gauss}|gauss={F64Bits(st.gauss)}"
                : "rsdict|" + RandomStateText(st);

        /// <summary>
        ///     Replaces the named <c>field=value</c> entries of a canonical state text by <c>field=*</c> — the generator's
        ///     <c>mask_text</c>, for the words an OS-entropy seeding draws (neither side can reproduce them).
        /// </summary>
        /// <param name="text">The state text.</param>
        /// <param name="fields">The fields to mask (empty: unchanged).</param>
        /// <returns>The masked text.</returns>
        internal static string MaskText(string text, IReadOnlyList<string> fields)
        {
            foreach (var f in fields)
                text = Regex.Replace(text, "(^|\\|)" + Regex.Escape(f) + "=[^|]*", "${1}" + f + "=*");
            return text;
        }

        /// <summary>
        ///     <see cref="MaskText"/> applied to every state text inside an observation tree (string values under the keys
        ///     <c>state</c>, <c>v</c> and <c>seed_seq</c>) — the generator's <c>mask_obs</c>. Mutates and returns the node.
        /// </summary>
        /// <param name="node">The observation.</param>
        /// <param name="fields">The fields to mask (empty: unchanged).</param>
        /// <returns>The masked observation.</returns>
        internal static JsonNode MaskObs(JsonNode node, IReadOnlyList<string> fields)
        {
            if (fields.Count == 0 || node is null)
                return node;
            if (node is JsonObject o)
            {
                foreach (var key in o.Select(kv => kv.Key).ToList())
                {
                    var child = o[key];
                    if ((key == "state" || key == "v" || key == "seed_seq") && child is JsonValue v && v.TryGetValue<string>(out var s))
                        o[key] = MaskText(s, fields);
                    else
                        MaskObs(child, fields);
                }
            }
            else if (node is JsonArray arr)
            {
                foreach (var x in arr)
                    MaskObs(x, fields);
            }
            return node;
        }

        /// <summary>
        ///     A C# primitive scalar as a 0-d NDArray of the matching dtype, or null when the value is not one of the scalar
        ///     types NumPy has a dtype for (big integers, text, objects).
        /// </summary>
        /// <param name="value">The scalar.</param>
        /// <returns>A new 0-d array (the caller disposes it), or null.</returns>
        private static NDArray TypedScalar(object value) => value switch
        {
            bool v => NDArray.Scalar(v),
            byte v => NDArray.Scalar(v),
            sbyte v => NDArray.Scalar(v),
            short v => NDArray.Scalar(v),
            ushort v => NDArray.Scalar(v),
            int v => NDArray.Scalar(v),
            uint v => NDArray.Scalar(v),
            long v => NDArray.Scalar(v),
            ulong v => NDArray.Scalar(v),
            Half v => NDArray.Scalar(v),
            float v => NDArray.Scalar(v),
            double v => NDArray.Scalar(v),
            Complex v => NDArray.Scalar(v),
            _ => null,
        };

        /// <summary>
        ///     An NDArray result: the scalar kind the corpus expects when the array is 0-d and of the implied dtype, else
        ///     the array observation.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="want">The expected observation kind.</param>
        /// <returns>The observation.</returns>
        private static JsonNode ObserveArray(NDArray nd, string want)
        {
            if (nd.ndim == 0)
            {
                if (want == "float" && nd.typecode == NPTypeCode.Double)
                    return FloatObs(nd.GetAtIndex<double>(0));
                if (want == "int" && IsInteger(nd.typecode))
                    return new JsonObject { ["k"] = "int", ["v"] = IntegerText(nd) };
                if (want == "bool" && nd.typecode == NPTypeCode.Boolean)
                    return new JsonObject { ["k"] = "bool", ["v"] = nd.GetAtIndex<bool>(0) };
            }
            return ArrayObs(nd);
        }

        /// <summary>The array observation: NumPy dtype name, shape, and the C-order bytes as lowercase hex.</summary>
        /// <param name="nd">The array (any layout; read in logical C order).</param>
        /// <returns>The observation.</returns>
        internal static JsonObject ArrayObs(NDArray nd)
        {
            var shape = new JsonArray();
            foreach (long d in nd.Shape.dimensions ?? Array.Empty<long>())
                shape.Add(d);
            return new JsonObject
            {
                ["k"] = "array",
                ["dtype"] = nd.typecode.AsNumpyDtypeName(),
                ["shape"] = shape,
                ["hex"] = Convert.ToHexString(FuzzCorpus.ResultBytes(nd)).ToLowerInvariant(),
            };
        }

        /// <summary>A C# scalar that NumPy would have returned as a numpy scalar: rendered through a 0-d array.</summary>
        /// <param name="scalar">The 0-d array (disposed here).</param>
        /// <returns>The array observation.</returns>
        private static JsonObject ArrayObsScalar(NDArray scalar)
        {
            using (scalar)
                return ArrayObs(scalar);
        }

        /// <summary>A Python-float observation: the float64 bit pattern.</summary>
        /// <param name="d">The value.</param>
        /// <returns>The observation.</returns>
        internal static JsonObject FloatObs(double d)
            => new() { ["k"] = "float", ["bits"] = F64Bits(d) };

        /// <summary>The IEEE-754 bits of a double as 16 lowercase hex digits (the generator's <c>f64_bits</c>).</summary>
        /// <param name="d">The value.</param>
        /// <returns>The hex text.</returns>
        internal static string F64Bits(double d)
            => unchecked((ulong)BitConverter.DoubleToInt64Bits(d)).ToString("x16", CultureInfo.InvariantCulture);

        /// <summary>Whether a type code is an integer type (not bool, not char).</summary>
        /// <param name="tc">The type code.</param>
        /// <returns>True for the eight integer widths.</returns>
        private static bool IsInteger(NPTypeCode tc)
            => tc is NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Int32
                or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;

        /// <summary>The decimal text of a 0-d integer array's value (unsigned read as unsigned).</summary>
        /// <param name="nd">A 0-d integer array.</param>
        /// <returns>The value's decimal text.</returns>
        private static string IntegerText(NDArray nd) => nd.typecode switch
        {
            NPTypeCode.UInt64 => nd.GetAtIndex<ulong>(0).ToString(CultureInfo.InvariantCulture),
            NPTypeCode.UInt32 => nd.GetAtIndex<uint>(0).ToString(CultureInfo.InvariantCulture),
            NPTypeCode.UInt16 => nd.GetAtIndex<ushort>(0).ToString(CultureInfo.InvariantCulture),
            NPTypeCode.Byte => nd.GetAtIndex<byte>(0).ToString(CultureInfo.InvariantCulture),
            NPTypeCode.SByte => nd.GetAtIndex<sbyte>(0).ToString(CultureInfo.InvariantCulture),
            NPTypeCode.Int16 => nd.GetAtIndex<short>(0).ToString(CultureInfo.InvariantCulture),
            NPTypeCode.Int32 => nd.GetAtIndex<int>(0).ToString(CultureInfo.InvariantCulture),
            _ => nd.GetAtIndex<long>(0).ToString(CultureInfo.InvariantCulture),
        };

        // ---------------------------------------------------------------------------------------------------------------
        // Receiver state
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     The receiver's canonical state text after the call — the generator's <c>recv_state_text</c>: a Generator's
        ///     bit generator, a RandomState's bit generator plus its Gaussian cache, a bit generator's own state; empty for
        ///     receivers without state.
        /// </summary>
        /// <param name="receiver">The receiver.</param>
        /// <returns>The canonical text ("" when the receiver has no state).</returns>
        internal static string StateText(object receiver) => receiver switch
        {
            Generator g => BitGeneratorStateText(g.bit_generator.state),
            NumPyRandom rs => RandomStateText((NumPyRandom.State)rs.get_state(false)),
            BitGenerator bg => BitGeneratorStateText(bg.state),
            // State-object receivers (their members read and write the object itself): the object's own text.
            NativeRandomState t => LegacyTupleText(t),
            NumPyRandom.State st => RsDictText(st),
            BitGeneratorState bgs => "bgstate|" + BitGeneratorStateText(bgs),
            ISeedSequence seq => SeedSeqText(seq),
            _ => "",
        };

        /// <summary>A RandomState's state text: its bit generator's, then <c>has_gauss</c> and the cached Gaussian's bits.</summary>
        /// <param name="st">The dict-form state (<c>get_state(legacy: false)</c>).</param>
        /// <returns>The canonical text.</returns>
        internal static string RandomStateText(NumPyRandom.State st)
            => BitGeneratorStateText(st.state) + $"|has_gauss={st.has_gauss}|gauss={F64Bits(st.gauss)}";

        /// <summary>
        ///     A bit generator state's canonical text, per engine, exactly as the generator's <c>bitgen_state_text</c> writes
        ///     it.
        /// </summary>
        /// <param name="state">The typed state.</param>
        /// <returns>The canonical text.</returns>
        /// <exception cref="NotSupportedException">A state type the observer does not know.</exception>
        internal static string BitGeneratorStateText(BitGeneratorState state) => state switch
        {
            // An unset array (a parameterless State) reads "null", as the generator's text does.
            MT19937.State mt => $"MT19937|pos={mt.pos}|key={(mt.key is null ? "null" : Sha256Hex(mt.key))}",
            PCG64.State p => $"PCG64|state={p.state}|inc={p.inc}|has_uint32={p.has_uint32}|uinteger={p.uinteger}",
            PCG64DXSM.State p => $"PCG64DXSM|state={p.state}|inc={p.inc}|has_uint32={p.has_uint32}|uinteger={p.uinteger}",
            Philox.State p => "Philox|counter=" + Join(p.counter) + "|key=" + Join(p.key) + "|buffer=" + Join(p.buffer) +
                              $"|buffer_pos={p.buffer_pos}|has_uint32={p.has_uint32}|uinteger={p.uinteger}",
            SFC64.State s => "SFC64|state=" + Join(s.state) + $"|has_uint32={s.has_uint32}|uinteger={s.uinteger}",
            _ => throw new NotSupportedException($"unknown bit generator state {state?.GetType().Name}"),
        };

        /// <summary>Comma-joined decimal words; <c>null</c> for an unset array (a parameterless State's).</summary>
        /// <param name="words">The words, or null.</param>
        /// <returns>The text.</returns>
        private static string Join(ulong[] words)
            => words is null ? "null" : string.Join(",", words.Select(w => w.ToString(CultureInfo.InvariantCulture)));

        /// <summary>The SHA-256 of 32-bit words laid out little-endian (NumPy's uint32 <c>tobytes()</c>), as lowercase hex.</summary>
        /// <param name="words">The words.</param>
        /// <returns>The hex digest.</returns>
        private static string Sha256Hex(uint[] words)
        {
            var bytes = new byte[words.Length * 4];
            for (int i = 0; i < words.Length; i++)
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4), words[i]);
            if (!BitConverter.IsLittleEndian)
                for (int i = 0; i < words.Length; i++)
                    Array.Reverse(bytes, i * 4, 4);
            return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        }

        // ---------------------------------------------------------------------------------------------------------------
        // Comparison
        // ---------------------------------------------------------------------------------------------------------------

        /// <summary>
        ///     Compares NumPy's observation with NumSharp's, structurally.
        /// </summary>
        /// <param name="expected">NumPy's observation.</param>
        /// <param name="actual">NumSharp's observation.</param>
        /// <param name="path">The location for the report (<c>result</c>, <c>result.items[2]</c>, …).</param>
        /// <returns>Null when equal, else a one-paragraph description of the first difference (for arrays: dtype, shape, or
        ///     the first differing element with both values).</returns>
        internal static string Diff(JsonElement expected, JsonNode actual, string path = "result")
        {
            string kind = expected.GetProperty("k").GetString();
            string gotKind = actual?["k"]?.GetValue<string>();
            if (kind != gotKind)
                return $"{path}: NumPy gives a {kind} ({Show(expected)}), NumSharp a {gotKind} ({Show(actual)})";
            switch (kind)
            {
                case "none":
                    return null;
                case "array":
                {
                    string dt = expected.GetProperty("dtype").GetString(), gotDt = actual["dtype"]!.GetValue<string>();
                    if (dt != gotDt)
                        return $"{path}: dtype {gotDt} != NumPy {dt}";
                    string sh = expected.GetProperty("shape").GetRawText(), gotSh = actual["shape"]!.ToJsonString();
                    if (Compact(sh) != Compact(gotSh))
                        return $"{path}: shape {gotSh} != NumPy {sh}";
                    string hex = expected.GetProperty("hex").GetString(), gotHex = actual["hex"]!.GetValue<string>();
                    if (hex == gotHex)
                        return null;
                    return $"{path}: values differ — " + FirstDifference(dt, hex, gotHex);
                }
                case "float":
                {
                    string bits = expected.GetProperty("bits").GetString(), got = actual["bits"]!.GetValue<string>();
                    return bits == got ? null
                        : $"{path}: {FromBits(got):R} (0x{got}) != NumPy {FromBits(bits):R} (0x{bits})";
                }
                case "int":
                case "text":
                {
                    string v = expected.GetProperty("v").GetString(), got = actual["v"]!.GetValue<string>();
                    return v == got ? null : $"{path}: {Quote(got)} != NumPy {Quote(v)}";
                }
                case "bool":
                {
                    bool v = expected.GetProperty("v").GetBoolean(), got = actual["v"]!.GetValue<bool>();
                    return v == got ? null : $"{path}: {got} != NumPy {v}";
                }
                case "bitgen":
                case "rs":
                case "gen":
                case "legacy_state":
                case "rsdict":
                case "bgstate":
                case "seedseq":
                    // Object observations are flat string fields (type, str, canonical state text): all must match.
                    foreach (var prop in expected.EnumerateObject())
                    {
                        if (prop.Name == "k")
                            continue;
                        string want = prop.Value.GetString(), got = actual[prop.Name]?.GetValue<string>();
                        if (want != got)
                            return $"{path}.{prop.Name}: {Quote(got)} != NumPy {Quote(want)}";
                    }
                    return null;
                case "seq":
                {
                    // Length first (a spawn of the wrong count), then item by item.
                    var want = expected.GetProperty("items");
                    var got = actual["items"]!.AsArray();
                    if (want.GetArrayLength() != got.Count)
                        return $"{path}: {got.Count} items != NumPy {want.GetArrayLength()}";
                    for (int i = 0; i < got.Count; i++)
                    {
                        string d = Diff(want[i], got[i], $"{path}.items[{i}]");
                        if (d != null)
                            return d;
                    }
                    return null;
                }
                default:
                    return $"{path}: unknown observation kind '{kind}'";
            }
        }

        /// <summary>The first differing element of two same-dtype buffers, both values shown (ULP distance for floats).</summary>
        /// <param name="dtype">The NumPy dtype name.</param>
        /// <param name="wantHex">NumPy's bytes.</param>
        /// <param name="gotHex">NumSharp's bytes.</param>
        /// <returns>The description.</returns>
        private static string FirstDifference(string dtype, string wantHex, string gotHex)
        {
            var want = FuzzCorpus.FromHex(wantHex);
            var got = FuzzCorpus.FromHex(gotHex);
            var tc = FuzzCorpus.DtypeToTC(dtype);
            var diffs = BitDiff.Compare(want, got, tc, nanBitExact: true);
            if (diffs.Count == 0)
                return "(byte difference with no element difference)";
            var d = diffs[0];
            string ulp = tc is NPTypeCode.Double or NPTypeCode.Single or NPTypeCode.Half
                ? $", {BitDiff.UlpDistance(want, got, d.Index, tc)} ULP"
                : "";
            return $"{diffs.Count} element(s); first at [{d.Index}]: NumSharp {d.Actual} vs NumPy {d.Expected}{ulp}";
        }

        /// <summary>A double from the generator's bit text.</summary>
        /// <param name="bits">16 hex digits.</param>
        /// <returns>The value.</returns>
        private static double FromBits(string bits)
            => BitConverter.Int64BitsToDouble(unchecked((long)ulong.Parse(bits, NumberStyles.HexNumber)));

        /// <summary>Strips JSON whitespace for a structural compare of small arrays.</summary>
        /// <param name="json">The JSON text.</param>
        /// <returns>The text without spaces.</returns>
        private static string Compact(string json) => json.Replace(" ", "", StringComparison.Ordinal);

        /// <summary>A length-capped rendering of an observation for a report.</summary>
        /// <param name="e">The observation.</param>
        /// <returns>The text.</returns>
        private static string Show(JsonElement e) => Cap(e.GetRawText());

        /// <summary>A length-capped rendering of an observation for a report.</summary>
        /// <param name="n">The observation.</param>
        /// <returns>The text.</returns>
        private static string Show(JsonNode n) => Cap(n?.ToJsonString() ?? "null");

        /// <summary>Caps a string at 240 characters for a failure line.</summary>
        /// <param name="s">The text.</param>
        /// <returns>The capped text.</returns>
        private static string Cap(string s) => s.Length <= 240 ? s : s.Substring(0, 240) + "…";

        /// <summary>Quotes a value for a failure line.</summary>
        /// <param name="s">The text.</param>
        /// <returns>The quoted text.</returns>
        private static string Quote(string s) => "\"" + Cap(s ?? "") + "\"";
    }
}
