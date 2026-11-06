using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using NumSharp;
using Tensorflow;
using Tensorflow.Eager;
using static Tensorflow.Binding;
using NSArray = NumSharp.NDArray;
using NSShape = NumSharp.Shape;
using TShape = Tensorflow.Shape;
using TFArray = Tensorflow.NumPy.NDArray;

namespace TfProbe
{
    /// <summary>
    ///     The probe: does TensorFlow.NET 0.150.0 (unchanged, from NuGet) work with NumSharp's text arrays from the
    ///     <c>strings</c> branch? Each check prints PASS / FAIL / INFO; the exit code is the number of FAILs.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Two routes are compared everywhere. The PUBLIC route uses only public API on both sides (what a TensorFlow.NET
    ///     user, or TensorFlow.NET itself, can write today): <see cref="NSArray.AsStringArray"/> / <c>GetAtIndex</c> on the
    ///     NumSharp side, <c>EagerTensor(string[])</c>, <c>StringTensor(byte[][])</c>, <c>StringData()</c> /
    ///     <c>StringBytes()</c> on the TensorFlow.NET side. The DIRECT route is <see cref="TfTextBridge"/>, which moves the
    ///     bytes between NumSharp's buffers and TensorFlow's <c>TF_TString</c> cells itself. Expected answers come from Python
    ///     TensorFlow 2.21 + NumPy 2.4.2 (<c>py_reference.py</c>, its JSON files in the data folder).
    ///     </para>
    ///     <para>
    ///     Usage: <c>NumSharp.DotNetRunScript.exe &lt;data dir&gt; [all|routes|ops|edges|lifetime|npy|timing-only] [n]</c>,
    ///     where the data dir is the one <c>py_reference.py</c> wrote and <c>n</c> the timing section's text count
    ///     (default 1,000,000). Set <c>TF_CPP_MIN_LOG_LEVEL=2</c> to silence TensorFlow's start-up log.
    ///     </para>
    /// </remarks>
    internal static class Program
    {
        private static int failures;

        // Tensors whose SafeStringTensorHandle another TensorFlow.NET object shares (the public bytes route): kept
        // reachable to the end so a finalizer cannot dealloc texts a later check still reads.
        private static readonly List<object> keepAlive = new List<object>();

        /// <summary>Print one check's outcome and count a failure.</summary>
        /// <param name="name">What was checked.</param>
        /// <param name="ok">Whether it held.</param>
        /// <param name="detail">What was seen, printed after the name.</param>
        private static void Check(string name, bool ok, string detail = null)
        {
            if (!ok)
                failures++;
            Console.WriteLine($"{(ok ? "PASS" : "FAIL")}  {name}{(detail is null ? "" : "  -- " + detail)}");
        }

        /// <summary>Print an observation that is not a pass/fail claim (an outcome being recorded).</summary>
        /// <param name="text">The observation.</param>
        private static void Info(string text) => Console.WriteLine($"INFO  {text}");

        /// <summary>Run <paramref name="action"/> and describe its outcome: its value, or the exception type and message.</summary>
        /// <param name="action">The call to observe.</param>
        /// <returns>"ok: ..." or "ExceptionType: message".</returns>
        private static string Outcome(Func<string> action)
        {
            try
            {
                return "ok: " + action();
            }
            catch (Exception e)
            {
                return $"{e.GetType().Name}: {e.Message.Split('\n')[0]}";
            }
        }

        /// <summary>Lowercase hex of a byte string (Python's <c>bytes.hex()</c>).</summary>
        private static string Hex(byte[] b) => Convert.ToHexString(b).ToLowerInvariant();

        /// <summary>Every element of a string tensor as hex, row-major.</summary>
        private static string[] TfHex(Tensor t) => t.StringBytes().Select(Hex).ToArray();

        /// <summary>
        ///     The bytes each element of a NumSharp text array should become in TensorFlow, in logical order, by NumPy's and
        ///     Python TF's rules: U and StringDType = the text's UTF-8; S = the bytes as NumPy reads them (trailing NULs
        ///     dropped).
        /// </summary>
        /// <param name="a">A text array.</param>
        /// <returns>The expected element bytes.</returns>
        private static byte[][] ExpectedBytes(NSArray a)
        {
            if (a.dtype.kind == 'S')
                return Enumerable.Range(0, (int)a.size).Select(i => (byte[])a.Unsafe.Storage.GetAtIndex(i)).ToArray();
            return NSArray.AsStringArray(a).Select(s => new UTF8Encoding(false, true).GetBytes(s)).ToArray();
        }

        /// <summary>Element-wise byte equality of two string-tensor-shaped byte lists.</summary>
        private static bool Same(byte[][] a, byte[][] b) => a.Length == b.Length && a.Zip(b).All(p => p.First.AsSpan().SequenceEqual(p.Second));

        /// <summary>The first differing element of two byte lists, for a FAIL's detail.</summary>
        private static string FirstDiff(byte[][] a, byte[][] b)
        {
            if (a.Length != b.Length)
                return $"length {a.Length} vs {b.Length}";
            for (int i = 0; i < a.Length; i++)
                if (!a[i].AsSpan().SequenceEqual(b[i]))
                    return $"element {i}: {Hex(a[i])} vs {Hex(b[i])}";
            return "equal";
        }

        // ------------------------------------------------------------------------------------------------------------
        // the public route: only public API of NumSharp and TensorFlow.NET
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>NumSharp text array to a TF string tensor through public API only.</summary>
        /// <param name="a">An S, U or StringDType array.</param>
        /// <returns>The tensor.</returns>
        private static Tensor PublicToTensor(NSArray a)
        {
            if (a.dtype.kind == 'S')
            {
                // NumSharp's public bytes of each element, then TensorFlow.NET's only multi-element bytes constructor (an
                // instance method of Tensor that uses no instance state).
                var items = new byte[a.size][];
                for (long i = 0; i < a.size; i++)
                    items[i] = (byte[])a.Unsafe.Storage.GetAtIndex(i);
                return PublicBytesTensor(items, a.shape);
            }
            // U and StringDType: managed strings (UTF-16), which TensorFlow.NET encodes to UTF-8 again.
            return new EagerTensor(NSArray.AsStringArray(a), new TShape(a.shape));
        }

        /// <summary>A TF string tensor from raw element bytes through TensorFlow.NET's public API.</summary>
        private static Tensor PublicBytesTensor(byte[][] items, long[] dims)
        {
            var holder = new Tensor(0);
            SafeStringTensorHandle h = holder.StringTensor(items, new TShape(dims));
            var plain = new Tensor(h);
            var eager = new TFArray(plain);
            keepAlive.Add(holder);
            keepAlive.Add(plain);
            return eager;
        }

        /// <summary>A TF string tensor to a NumSharp array through public API only.</summary>
        /// <param name="t">The tensor.</param>
        /// <param name="kind">'U', 'T' or 'S'.</param>
        /// <returns>The array, reshaped to the tensor's shape.</returns>
        private static NSArray PublicToNDArray(Tensor t, char kind)
        {
            long[] dims = t.shape.dims;
            NSArray flat;
            if (kind == 'S')
            {
                byte[][] b = t.StringBytes();
                int w = Math.Max(1, b.Length == 0 ? 1 : b.Max(x => x.Length));
                flat = np.array(b.Cast<object>().ToArray(), np.dtype("S" + w));
            }
            else
            {
                string[] s = t.StringData();
                flat = kind == 'U' ? np.array(s) : np.array(s.Cast<object>().ToArray(), np.dtypes.StringDType());
            }
            return dims.Length == 1 ? flat : flat.reshape(new NSShape(dims));
        }

        private static int Main(string[] args)
        {
            string dataDir = args[0];
            string section = args.Length > 1 ? args[1] : "all";
            Console.OutputEncoding = Encoding.UTF8;
            Info($"TensorFlow native {tf.VERSION}, TensorFlow.NET {typeof(Tensor).Assembly.GetName().Version}, NumSharp {typeof(NSArray).Assembly.GetName().Version}, eager={tf.executing_eagerly()}");

            string[] texts = JsonSerializer.Deserialize<string[]>(File.ReadAllText(Path.Combine(dataDir, "texts.json")));
            using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(dataDir, "reference.json")));

            if (section is "all" or "routes") Routes(texts);
            if (section is "all" or "ops") Ops(texts, reference.RootElement);
            if (section is "all" or "edges") Edges();
            if (section is "all" or "lifetime") Lifetime();
            if (section is "all" or "npy") Npy(dataDir);
            if (section is "all" or "timing" or "timing-only") Timing(args.Length > 2 ? int.Parse(args[2]) : 1_000_000);

            Console.WriteLine($"FAILURES {failures}");
            GC.KeepAlive(keepAlive);
            return failures;
        }

        // ------------------------------------------------------------------------------------------------------------
        // 1. both routes, every text dtype, several layouts
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Every text dtype and layout through both routes, NumSharp to TF and back.</summary>
        private static void Routes(string[] texts)
        {
            Console.WriteLine("== routes: NumSharp text -> TF string tensor -> NumSharp");
            using NSArray u = np.array(texts);                                                    // U, 1-D
            using NSArray t = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType());  // StringDType, 1-D
            using NSArray s = np.strings.encode(u);                                               // S (UTF-8), 1-D
            using NSArray u2 = np.array(texts.Take(12).ToArray()).reshape(3, 4);
            using NSArray t2 = np.array(texts.Take(12).Cast<object>().ToArray(), np.dtypes.StringDType()).reshape(3, 4);
            using NSArray s2 = np.strings.encode(u2);

            var cases = new (string name, NSArray a)[]
            {
                ("U 1-D", u), ("StringDType 1-D", t), ("S 1-D", s),
                ("U 3x4", u2), ("StringDType 3x4", t2), ("S 3x4", s2),
                ("U 3x4 transposed", u2.T), ("StringDType 3x4 transposed", t2.T), ("S 3x4 transposed", s2.T),
                ("U sliced [1:, ::2]", u2["1:, ::2"]), ("StringDType sliced [1:, ::2]", t2["1:, ::2"]), ("S sliced [1:, ::2]", s2["1:, ::2"]),
                ("U reversed", u["::-1"]), ("StringDType reversed", t["::-1"]), ("S reversed", s["::-1"]),
                ("U contiguous offset view [3:]", u["3:"]), ("StringDType contiguous offset view [3:]", t["3:"]), ("S contiguous offset view [3:]", s["3:"]),
            };

            foreach (var (name, a) in cases)
            {
                byte[][] expected = ExpectedBytes(a);
                Tensor pub = PublicToTensor(a);
                Check($"public  export {name}: bytes", Same(pub.StringBytes(), expected), FirstDiff(pub.StringBytes(), expected));
                Check($"public  export {name}: shape", pub.shape.dims.SequenceEqual(a.shape), $"{pub.shape} vs ({string.Join(",", a.shape)})");

                EagerTensor direct = TfTextBridge.ToTensor(a);
                Check($"direct  export {name}: bytes", Same(direct.StringBytes(), expected), FirstDiff(direct.StringBytes(), expected));
                Check($"direct  export {name}: shape", direct.shape.dims.SequenceEqual(a.shape), $"{direct.shape}");

                if (a.dtype.kind != 'U')
                {
                    EagerTensor view = TfTextBridge.ToTensor(a, TfTextBridge.Payload.View);
                    Check($"view    export {name}: bytes", Same(view.StringBytes(), expected), FirstDiff(view.StringBytes(), expected));
                }

                // Back: the direct import into the class that keeps every byte (S -> S, U / StringDType -> StringDType).
                using NSArray back = TfTextBridge.ToNDArray(direct, a.dtype.kind == 'S' ? TfTextBridge.TextKind.Bytes : TfTextBridge.TextKind.StringDType);
                Check($"direct  import {name}: same elements", Same(ExpectedBytes(back), expected) && back.shape.SequenceEqual(a.shape),
                    $"{back.dtype} shape ({string.Join(",", back.shape)})");
                using NSArray backPublic = PublicToNDArray(pub, a.dtype.kind == 'S' ? 'S' : 'T');
                Check($"public  import {name}: same elements", Same(ExpectedBytes(backPublic), expected), $"{backPublic.dtype}");
            }

            // StringDType texts in all three storage places: short (in the cell), arena (16+ bytes), heap (a longer
            // text written over an arena text after the array was built).
            using NSArray places = np.array(new object[] { "short", "an arena text of more than fifteen bytes", "x", "another arena text, also longer" }, np.dtypes.StringDType());
            places[2] = "a heap text: written after construction and longer than any arena slot it could reuse here";
            places[1] = "now an even longer text that cannot fit the old arena allocation, so it moves to the heap";
            byte[][] want = ExpectedBytes(places);
            EagerTensor pl = TfTextBridge.ToTensor(places);
            Check("direct  export StringDType short/arena/heap texts", Same(pl.StringBytes(), want), FirstDiff(pl.StringBytes(), want));
            EagerTensor plv = TfTextBridge.ToTensor(places, TfTextBridge.Payload.View);
            Check("view    export StringDType short/arena/heap texts", Same(plv.StringBytes(), want), FirstDiff(plv.StringBytes(), want));
            using NSArray plBack = TfTextBridge.ToNDArray(pl);
            Check("direct  import StringDType short/arena/heap texts", Same(ExpectedBytes(plBack), want));
        }

        // ------------------------------------------------------------------------------------------------------------
        // 2. TF's string ops on NumSharp-made tensors vs Python TF, and NumSharp's np.strings vs NumPy
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>TensorFlow.NET's tf.strings ops over a NumSharp-made tensor, against Python TF's and NumPy's answers.</summary>
        private static void Ops(string[] texts, JsonElement r)
        {
            Console.WriteLine("== ops: TF.NET tf.strings over NumSharp-made tensors vs Python TF 2.21; NumSharp np.strings vs NumPy 2.4.2");
            using NSArray u = np.array(texts);
            EagerTensor tt = TfTextBridge.ToTensor(u);

            string[] Ref(string key) => r.GetProperty(key).EnumerateArray().Select(e => e.GetString()).ToArray();
            int[] RefInts(string key) => r.GetProperty(key).EnumerateArray().Select(e => e.GetInt32()).ToArray();
            void Compare(string name, string[] got, string[] want)
            {
                var bad = Enumerable.Range(0, want.Length).Where(i => got[i] != want[i]).ToArray();
                Check(name, bad.Length == 0, bad.Length == 0 ? $"{want.Length}/{want.Length}" : string.Join("; ", bad.Select(i => $"[{i}] {got[i]} vs {want[i]}")));
            }

            Compare("tf.strings.string_length(BYTE) == Python TF", tf.strings.string_length(tt).ToArray<int>().Select(x => x.ToString()).ToArray(), RefInts("tf_length_byte").Select(x => x.ToString()).ToArray());
            Compare("tf.strings.string_length(UTF8_CHAR) == Python TF", tf.strings.string_length(tt, unit: "UTF8_CHAR").ToArray<int>().Select(x => x.ToString()).ToArray(), RefInts("tf_length_utf8").Select(x => x.ToString()).ToArray());
            Compare("tf.strings.lower(ascii) == Python TF", TfHex(tf.strings.lower(tt)), Ref("tf_lower_ascii"));
            // TF.NET's lower(encoding: "utf-8") is compared with BOTH Python answers: equal to the ASCII one means the
            // encoding never reached the op. The raw op with the attribute set is the control.
            string[] netUtf8 = TfHex(tf.strings.lower(tt, encoding: "utf-8"));
            int vsUtf8 = Enumerable.Range(0, texts.Length).Count(i => netUtf8[i] != Ref("tf_lower_utf8")[i]);
            int vsAscii = Enumerable.Range(0, texts.Length).Count(i => netUtf8[i] != Ref("tf_lower_ascii")[i]);
            Info($"TF.NET tf.strings.lower(encoding: \"utf-8\"): differs from Python's utf-8 lower on {vsUtf8}/{texts.Length}, from Python's ASCII lower on {vsAscii}/{texts.Length}");
            Tensor raw = tf.Context.ExecuteOp("StringLower", null, new ExecuteOpArgs(tt).SetAttributes(new { encoding = "utf-8" }));
            Compare("raw StringLower op with the encoding ATTRIBUTE set == Python TF utf-8 lower", TfHex(raw), Ref("tf_lower_utf8"));
            Compare("tf.strings.substr(0, 2, UTF8_CHAR) == Python TF", TfHex(tf.strings.substr(tt, 0, 2, @uint: "UTF8_CHAR")), Ref("tf_substr_utf8_0_2"));
            Compare("tf.strings.substr(0, 1, BYTE) == Python TF", TfHex(tf.strings.substr(tt, 0, 1)), Ref("tf_substr_byte_0_1"));
            Compare("tf.strings.regex_replace == Python TF", TfHex(tf.strings.regex_replace(tt, "[aeiou]", "_")), Ref("tf_regex_replace"));
            Info("TF.NET has no tf.strings.upper (Python TF does): " + Outcome(() => typeof(tensorflow.StringsApi).GetMethod("upper") is null ? "method absent" : "present"));

            string[] Utf8Hex(NSArray a) => NSArray.AsStringArray(a).Select(x => Hex(Encoding.UTF8.GetBytes(x))).ToArray();
            using (NSArray n = np.strings.str_len(u))
                Compare("np.strings.str_len == NumPy", n.ToArray<long>().Select(x => x.ToString()).ToArray(), RefInts("np_str_len").Select(x => x.ToString()).ToArray());
            using (NSArray n = np.strings.lower(u)) Compare("np.strings.lower == NumPy", Utf8Hex(n), Ref("np_lower"));
            using (NSArray n = np.strings.upper(u)) Compare("np.strings.upper == NumPy", Utf8Hex(n), Ref("np_upper"));
            using (NSArray n = np.strings.slice(u, 0, 2)) Compare("np.strings.slice(0, 2) == NumPy", Utf8Hex(n), Ref("np_slice_0_2"));

            // The CPU fill-in for TF.NET's missing upper: NumSharp's np.strings.upper over the tensor's text, back to TF.
            // It must give NumPy's answer; where Python TF's own upper(utf-8) differs, that is printed, not failed.
            using NSArray asT = TfTextBridge.ToNDArray(tt);
            using NSArray upT = np.strings.upper(asT);
            EagerTensor upTf = TfTextBridge.ToTensor(upT);
            string[] upHex = TfHex(upTf);
            Compare("np.strings.upper over a TF tensor (StringDType round trip) == NumPy np.strings.upper", upHex, Ref("np_upper"));
            var tfDiff = Enumerable.Range(0, texts.Length).Where(i => upHex[i] != Ref("tf_upper_utf8")[i]).ToArray();
            Info($"...and vs Python tf.strings.upper(utf-8): {texts.Length - tfDiff.Length}/{texts.Length} equal" +
                 string.Concat(tfDiff.Select(i => $"; [{i}] NumSharp {upHex[i]} vs TF {Ref("tf_upper_utf8")[i]}")));
        }

        // ------------------------------------------------------------------------------------------------------------
        // 3. edge cases: 0-d, empty, NA, lone surrogate, invalid UTF-8, NULs
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>The edges where the two libraries' text models differ.</summary>
        private static void Edges()
        {
            Console.WriteLine("== edges");
            // 0-d both ways.
            using (NSArray scalar = NSArray.Scalar("scalar"))
            {
                EagerTensor t0 = TfTextBridge.ToTensor(scalar);
                Check("0-d U -> TF scalar", t0.shape.ndim == 0 && t0.StringData()[0] == "scalar", $"{t0.shape} {t0.StringData()[0]}");
                using NSArray back0 = TfTextBridge.ToNDArray(t0);
                Check("TF scalar -> 0-d StringDType", back0.ndim == 0 && NSArray.AsStringArray(back0)[0] == "scalar", $"ndim {back0.ndim}");
                Info("public 0-d U -> TF: " + Outcome(() => PublicToTensor(scalar).shape.ToString()));
            }
            // Empty both ways.
            foreach (long[] dims in new[] { new long[] { 0 }, new long[] { 2, 0 } })
            {
                using NSArray e = np.empty(new NSShape(dims), np.dtypes.StringDType());
                EagerTensor te = TfTextBridge.ToTensor(e);
                Check($"empty StringDType {string.Join("x", dims)} -> TF", te.shape.dims.SequenceEqual(dims), te.shape.ToString());
                using NSArray eb = TfTextBridge.ToNDArray(te, TfTextBridge.TextKind.Bytes);
                Check($"empty TF {string.Join("x", dims)} -> S", eb.shape.SequenceEqual(dims), eb.dtype.ToString());
            }
            // NA: TensorFlow has no missing string.
            using (NSArray na = np.array(new object[] { "x", null }, np.dtypes.StringDType((object)null)))
            {
                Info("StringDType with an NA, direct route: " + Outcome(() => TfTextBridge.ToTensor(na).ToString()));
                Info("StringDType with an NA, public route: " + Outcome(() => string.Join("|", PublicToTensor(na).StringData())));
                Info("Python TF for a StringDType array (any NA or none): ValueError 'Unsupported numpy type' (conversions.json)");
            }
            // A lone surrogate in a U element.
            using (NSArray lone = np.array(new[] { "ok", "bad" + (char)0xD800 }))
            {
                Info("U with a lone surrogate, direct route: " + Outcome(() => TfTextBridge.ToTensor(lone).ToString()));
                Info("U with a lone surrogate, public route: " + Outcome(() => string.Join("|", PublicToTensor(lone).StringBytes().Select(Hex))));
                Info("Python TF: ValueError 'Unable to convert element to UTF-8'");
            }
            // Invalid UTF-8 produced by TF itself (a BYTE substr through a two-byte character).
            EagerTensor cut = (EagerTensor)tf.strings.substr(TfTextBridge.ToTensor(np.array(new[] { "été" })), 0, 1);
            Info("TF BYTE substr of 'été' -> StringDType, direct: " + Outcome(() => NSArray.AsStringArray(TfTextBridge.ToNDArray(cut))[0]));
            Info("TF BYTE substr of 'été' -> S, direct: " + Outcome(() => Hex((byte[])TfTextBridge.ToNDArray(cut, TfTextBridge.TextKind.Bytes).Unsafe.Storage.GetAtIndex(0))));
            Info("TF BYTE substr of 'été' -> public StringData(): " + Outcome(() => Hex(Encoding.UTF8.GetBytes(cut.StringData()[0]))));
            Info("Python: .numpy().astype(StringDType()) -> UnicodeDecodeError ''utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data'");

            // NULs: trailing and embedded, starting in TF (bytes).
            Tensor nul = PublicBytesTensor(new[] { new byte[] { 0x61, 0x62, 0 }, new byte[] { 0x61, 0, 0x62 } }, new long[] { 2 });
            using (NSArray ns = TfTextBridge.ToNDArray(nul, TfTextBridge.TextKind.Bytes))
                Info($"TF [b'ab\\0', b'a\\0b'] -> S: dtype {ns.dtype}, elements {string.Join(",", ExpectedBytes(ns).Select(Hex))} (Python .astype('S'): [b'ab', b'a\\x00b'], |S3)");
            using (NSArray nt = TfTextBridge.ToNDArray(nul))
                Info($"TF [b'ab\\0', b'a\\0b'] -> StringDType: elements {string.Join(",", ExpectedBytes(nt).Select(Hex))} (Python .astype(StringDType()) keeps both NULs)");
        }

        // ------------------------------------------------------------------------------------------------------------
        // 4. lifetime: who frees the texts, and when
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Collect until finalizers have run twice (TensorFlow.NET releases native handles from finalizers).</summary>
        private static void Settle()
        {
            for (int i = 0; i < 3; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }

        /// <summary>The ownership checks: view pins, deallocator counts, and a tensor that shares the buffer.</summary>
        private static void Lifetime()
        {
            Console.WriteLine("== lifetime");
            // The counters are process-wide: finalize the tensors earlier sections dropped (their view exports release
            // pins too) before the snapshots, so each check below counts its own tensor only.
            Settle();
            string[] longTexts = Enumerable.Range(0, 64).Select(i => new string((char)('a' + i % 26), 30 + i)).ToArray();

            // (a) A view export keeps the NumSharp buffer alive after the array is disposed, and gives it back with the tensor.
            long pins0 = TfTextBridge.ReleasedPins;
            EagerTensor view = MakeViewAndDisposeSource(longTexts);
            Settle();
            Check("view export reads right after the NumSharp array was disposed", view.StringData().SequenceEqual(longTexts));
            view = null;
            Settle();
            Check("view export: the pin is released when the tensor goes", TfTextBridge.ReleasedPins == pins0 + 1, $"released pins {TfTextBridge.ReleasedPins - pins0}");

            // (b) The direct copy export's texts live while ANY tensor shares the buffer (tf.identity forwards it).
            long blocks0 = TfTextBridge.ReleasedBlocks;
            Tensor y = IdentityOfDropped(useBridge: true, longTexts);
            Settle();
            Check("direct export: tf.identity's output still reads the texts after the source tensor is gone", y.StringData().SequenceEqual(longTexts),
                $"first element now '{y.StringData()[0]}'");
            y = null;
            Settle();
            Check("direct export: the block is freed with the last tensor", TfTextBridge.ReleasedBlocks > blocks0, $"freed {TfTextBridge.ReleasedBlocks - blocks0}");

            // (c) TensorFlow.NET's own string tensors (tf.constant(string[])) under the same sequence.
            Tensor y2 = IdentityOfDropped(useBridge: false, longTexts);
            Settle();
            string[] seen = y2.StringData();
            int lost = seen.Zip(longTexts).Count(p => p.First != p.Second);
            Info($"TF.NET tf.constant(string[]) + tf.identity, source dropped: {lost}/{longTexts.Length} texts changed (first now '{seen[0]}')");

            // (d) The control for (c): the same, but the source tensor stays reachable until after the read.
            Tensor x3 = tf.constant(longTexts);
            Tensor y3 = tf.identity(x3);
            Settle();
            int lost3 = y3.StringData().Zip(longTexts).Count(p => p.First != p.Second);
            GC.KeepAlive(x3);
            Info($"control: same, source kept alive: {lost3}/{longTexts.Length} texts changed");

            // (e) And short texts (22 bytes or fewer live inside the TF_TString cell, which TF_StringDealloc leaves alone).
            string[] shortTexts = Enumerable.Range(0, 64).Select(i => "t" + i).ToArray();
            Tensor y4 = tf.identity(DropAndReturn(shortTexts));
            Settle();
            Info($"short texts (in-cell), source dropped: {y4.StringData().Zip(shortTexts).Count(p => p.First != p.Second)}/{shortTexts.Length} changed");
        }

        /// <summary>A TF.NET string constant that nothing else references once the caller has used it.</summary>
        private static Tensor DropAndReturn(string[] texts) => tf.constant(texts);

        /// <summary>Build a StringDType array, view-export it, dispose the array, return the tensor.</summary>
        private static EagerTensor MakeViewAndDisposeSource(string[] texts)
        {
            NSArray a = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType());
            EagerTensor t = TfTextBridge.ToTensor(a, TfTextBridge.Payload.View);
            a.Dispose();
            return t;
        }

        /// <summary>Make a string tensor (bridge or TF.NET's own), take tf.identity of it, and drop the source.</summary>
        private static Tensor IdentityOfDropped(bool useBridge, string[] texts)
        {
            Tensor x;
            if (useBridge)
            {
                using NSArray a = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType());
                x = TfTextBridge.ToTensor(a);
            }
            else
                x = tf.constant(texts);
            return tf.identity(x);
        }

        // ------------------------------------------------------------------------------------------------------------
        // 5. NumPy-written .npy text files
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Load NumPy-written text .npy files with NumSharp and with TensorFlow.NET's own np.load.</summary>
        private static void Npy(string dataDir)
        {
            Console.WriteLine("== .npy files written by NumPy 2.4.2");
            foreach (string name in new[] { "u.npy", "s.npy", "t.npy", "tna.npy" })
            {
                string path = Path.Combine(dataDir, name);
                Info($"{name}: NumSharp np.load -> " + Outcome(() =>
                {
                    var a = (NSArray)np.load(path, allow_pickle: true);
                    string cross = Outcome(() => string.Join(",", TfTextBridge.ToTensor(a).StringBytes().Select(Hex)));
                    return $"{a.dtype} shape ({string.Join(",", a.shape)}); to TF: {cross}";
                }));
                Info($"{name}: TF.NET np.load   -> " + Outcome(() => Tensorflow.NumPy.np.load(path).ToString()));
            }
        }

        // ------------------------------------------------------------------------------------------------------------
        // 6. timing
        // ------------------------------------------------------------------------------------------------------------

        /// <summary>Time both routes over N random texts (lengths 1-40 code points, some non-ASCII and astral).</summary>
        private static void Timing(int n)
        {
            Console.WriteLine($"== timing, {n:N0} texts (best of 5 after 2 warm-ups; ms and ns/element)");
            var rng = new Random(12345);
            int[] palette = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ".Select(c => (int)c)
                .Concat(new[] { 0xE9, 0xDF, 0x65E5, 0x1F600 }).ToArray();
            var texts = new string[n];
            var sb = new StringBuilder();
            for (int i = 0; i < n; i++)
            {
                sb.Clear();
                int len = rng.Next(1, 41);
                for (int k = 0; k < len; k++)
                    sb.Append(char.ConvertFromUtf32(rng.Next(10) == 0 ? palette[rng.Next(palette.Length)] : palette[rng.Next(63)]));
                texts[i] = sb.ToString();
            }
            using NSArray u = np.array(texts);
            using NSArray t = np.array(texts.Cast<object>().ToArray(), np.dtypes.StringDType());
            using NSArray s = np.strings.encode(u);
            Info($"U itemsize {u.dtype.itemsize} B ({u.size * u.dtype.itemsize / 1e6:F0} MB), S itemsize {s.dtype.itemsize} B, StringDType cells {t.size * 16 / 1e6:F0} MB");

            void Time(string label, Func<object> run)
            {
                for (int w = 0; w < 2; w++)
                    Drop(run());
                double best = double.MaxValue;
                for (int r = 0; r < 5; r++)
                {
                    Settle();
                    var sw = Stopwatch.StartNew();
                    object result = run();
                    sw.Stop();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
                    Drop(result);
                }
                Console.WriteLine($"TIME  {label,-58} {best,9:F1} ms  {best * 1e6 / n,7:F1} ns/elem");
            }

            Time("export U   public (AsStringArray + EagerTensor(string[]))", () => PublicToTensor(u));
            Time("export U   direct (UCS-4 -> UTF-8 into TF_TString)", () => TfTextBridge.ToTensor(u));
            Time("export T   public (AsStringArray + EagerTensor(string[]))", () => PublicToTensor(t));
            Time("export T   direct copy (NpyString_load -> TF_StringCopy)", () => TfTextBridge.ToTensor(t));
            Time("export T   direct view (TF_TSTR_VIEW into the arena)", () => TfTextBridge.ToTensor(t, TfTextBridge.Payload.View));
            Time("export S   public (GetAtIndex byte[] + StringTensor(byte[][]))", () => PublicToTensor(s));
            Time("export S   direct copy", () => TfTextBridge.ToTensor(s));
            Time("export S   direct view", () => TfTextBridge.ToTensor(s, TfTextBridge.Payload.View));

            EagerTensor source = TfTextBridge.ToTensor(t);
            Time("import ->U public (StringData + np.array(string[]))", () => PublicToNDArray(source, 'U'));
            Time("import ->T public (StringData + np.array(object[], T))", () => PublicToNDArray(source, 'T'));
            Time("import ->T direct (UTF-8 check + NpyString_pack)", () => TfTextBridge.ToNDArray(source));
            Time("import ->S public (StringBytes + np.array(byte[][], S))", () => PublicToNDArray(source, 'S'));
            Time("import ->S direct (memcpy into S{w})", () => TfTextBridge.ToNDArray(source, TfTextBridge.TextKind.Bytes));
            GC.KeepAlive(source);
        }

        /// <summary>Dispose a timed result (an NDArray is disposed; TF.NET tensors are left to their finalizers, as TF.NET does).</summary>
        private static void Drop(object result)
        {
            if (result is NSArray a)
                a.Dispose();
        }
    }
}
