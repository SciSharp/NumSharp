using System;
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NumSharp.Tests.Fuzz.RandomApi
{
    /// <summary>
    ///     The documented, INTENDED divergences of the random-API oracle: each is keyed on the member AND a bound or a
    ///     condition, so a difference outside it still fails, and every excused case is printed by the replay (never
    ///     silent). Anything that is not listed here is a NumSharp bug to fix.
    /// </summary>
    /// <remarks>
    ///     The table mirrors plan §7 (docs/plans/random-oracle-coverage.md). Entries:
    ///     <list type="bullet">
    ///         <item><b>Generator.pareto / Generator.power</b> — NumPy's modern samplers are <c>expm1(E/a)</c> and
    ///             <c>pow(-expm1(-E), 1/a)</c>, and the win-amd64 CRT's <c>expm1</c> is a closed approximation inside
    ///             <c>[-ln 2, ln 1.5]</c>; NumSharp's Sun <c>s_expm1</c> is within 2 ULP there. pareto carries that bound
    ///             straight through; power's <c>pow(q, 1/a)</c> scales it to <c>ceil(2/a) + 3</c> ULP (the bound the
    ///             <c>grnd</c> tier already documents). Stream positions are identical: only the float64 values may move,
    ///             element by element, within the bound.</item>
    ///     </list>
    /// </remarks>
    internal static class RandomApiDivergences
    {
        /// <summary>
        ///     Classifies a value divergence of a random-API case.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <param name="expected">NumPy's result observation.</param>
        /// <param name="actual">NumSharp's result observation.</param>
        /// <returns>The documented reason when the divergence is an intended one inside its bound, else null.</returns>
        internal static string ClassifyValue(FuzzCorpus.Case c, JsonElement expected, JsonNode actual)
        {
            string member = c.Params["member"].GetString();
            if (member != "Generator.pareto" && member != "Generator.power")
                return null;
            string kind = expected.GetProperty("k").GetString();
            if (kind != actual?["k"]?.GetValue<string>())
                return null;

            byte[] want, got;
            if (kind == "array")
            {
                // Only the float64 values may move: a dtype or shape difference is a real divergence, never excused.
                if (expected.GetProperty("dtype").GetString() != "float64" || actual["dtype"]!.GetValue<string>() != "float64"
                    || expected.GetProperty("shape").GetRawText().Replace(" ", "") != actual["shape"]!.ToJsonString())
                    return null;
                want = FuzzCorpus.FromHex(expected.GetProperty("hex").GetString());
                got = FuzzCorpus.FromHex(actual["hex"]!.GetValue<string>());
            }
            else if (kind == "float")
            {
                // A size=None draw returns a Python float: the same element-wise bound applies to its one value, so it is
                // laid out as a one-element float64 buffer and judged by the same ULP rule as an array element.
                want = BitConverter.GetBytes(ulong.Parse(expected.GetProperty("bits").GetString(), NumberStyles.HexNumber));
                got = BitConverter.GetBytes(ulong.Parse(actual["bits"]!.GetValue<string>(), NumberStyles.HexNumber));
            }
            else
                return null;

            int bound = member == "Generator.pareto" ? 2 : PowerBound(c);
            if (bound <= 0)
                return null;
            var diffs = BitDiff.Compare(want, got, NPTypeCode.Double, nanBitExact: true);
            foreach (var d in diffs)
                if (!BitDiff.WithinUlp(want, got, d.Index, NPTypeCode.Double, bound))
                    return null;
            return $"{member}: in-band expm1 (Sun's s_expm1) vs the closed ucrtbase expm1, <= {bound} ULP [documented]";
        }

        /// <summary>
        ///     Classifies a NumSharp exception on a case where NumPy returns a value — only where the C# signature cannot
        ///     express NumPy's answer and NumSharp raises a deliberate, specific error instead.
        /// </summary>
        /// <param name="c">The case.</param>
        /// <param name="e">NumSharp's exception.</param>
        /// <returns>The documented reason, or null (the throw is a failure).</returns>
        /// <remarks>
        ///     <b>NumPyRandom.get_state()</b> on a RandomState whose engine is not MT19937: NumPy warns and returns the dict
        ///     form; the C# overload is typed as the legacy tuple (<see cref="NativeRandomState"/>), which cannot carry it,
        ///     so NumSharp raises a <see cref="ValueError"/> naming <c>get_state(legacy: false)</c> — the overload that
        ///     returns NumPy's dict (itself oracle-checked). Keyed on the signature, the exception type and its text, and
        ///     on NumPy's answer being the dict.
        /// </remarks>
        internal static string ClassifyThrow(FuzzCorpus.Case c, Exception e)
        {
            string sig = c.Params["sig"].GetString();
            if (sig == "NumPyRandom.get_state()" && e is ValueError
                && e.Message.StartsWith("get_state and legacy can only be used with the MT19937 BitGenerator.", StringComparison.Ordinal)
                && c.Expected.Result.ValueKind == JsonValueKind.Object && c.Expected.Result.GetProperty("k").GetString() == "rsdict")
                return "NumPyRandom.get_state(): the tuple-typed overload refuses a non-MT19937 engine (NumPy warns and returns the dict; " +
                       "get_state(legacy: false) is that dict) [documented]";
            // An MT19937 position outside [0, 624]: NumPy stores the C int unchecked, and its next draw indexes the 624-word
            // key with it (a read past the array); NumSharp refuses the state. Keyed on NumSharp's own refusal text, which
            // names the position, and only for a position that really is out of range.
            const string posPrefix = "state['state']['pos'] must be in [0, 624], got ";
            if (e is ValueError && e.Message.StartsWith(posPrefix, StringComparison.Ordinal)
                && long.TryParse(e.Message.AsSpan(posPrefix.Length), NumberStyles.Integer, CultureInfo.InvariantCulture, out long pos)
                && (pos < 0 || pos > 624))
                return "MT19937 position outside [0, 624]: NumPy stores it and its next draw reads past the key; NumSharp refuses [documented]";
            return null;
        }

        /// <summary>
        ///     Generator.power's bound, <c>ceil(2 / a) + 3</c> ULP, for the smallest exponent <c>a</c> the case draws with
        ///     (a scalar argument, or the minimum of an NDArray argument's elements). NaN elements never diverge and are
        ///     skipped.
        /// </summary>
        /// <param name="c">A Generator.power case.</param>
        /// <returns>The bound, or -1 when no positive finite <c>a</c> can be read.</returns>
        private static int PowerBound(FuzzCorpus.Case c)
        {
            double minA = double.PositiveInfinity;
            foreach (var arg in c.Params["args"].EnumerateArray())
            {
                if (arg.GetProperty("n").GetString() != "a" || arg.TryGetProperty("omit", out _) || arg.TryGetProperty("null", out _))
                    continue;
                if (arg.TryGetProperty("bits", out var bits))
                    minA = BitConverter.Int64BitsToDouble(unchecked((long)ulong.Parse(bits.GetString(), NumberStyles.HexNumber)));
                else if (arg.TryGetProperty("op", out var op))
                {
                    using var a = FuzzCorpus.Reconstruct(c.Operands[op.GetInt32()]);
                    using var ad = a.astype(np.float64);
                    for (long e = 0; e < ad.size; e++)
                    {
                        double v = ad.GetAtIndex<double>(e);
                        if (v < minA)
                            minA = v;
                    }
                }
            }
            return minA > 0 && double.IsFinite(2.0 / minA) ? (int)Math.Ceiling(2.0 / minA) + 3 : -1;
        }
    }
}
