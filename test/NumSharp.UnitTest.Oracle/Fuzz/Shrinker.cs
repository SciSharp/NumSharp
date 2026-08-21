using System;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using NumSharp;

namespace NumSharp.UnitTest.Fuzz
{
    /// <summary>
    ///     Minimizes a failing element-wise case to a single-element reproduction, so a divergence
    ///     found in a large random/soak case becomes a tiny, committable regression case. For an
    ///     element-wise op, the output element at <c>diffIndex</c> depends only on the corresponding
    ///     input element(s), so we can carve out a 1-element case using the corpus's own expected
    ///     byte for that index — no NumPy needed. Broadcast operands that are neither scalar nor the
    ///     full output size are left unshrunk (returns null).
    /// </summary>
    public static class Shrinker
    {
        private static readonly HashSet<string> ElementwiseOps = new()
        {
            "astype", "add", "subtract", "multiply", "divide", "floor_divide", "mod", "power",
            "equal", "not_equal", "less", "greater", "less_equal", "greater_equal",
            "negative", "abs", "sign", "sqrt", "cbrt", "square", "reciprocal",
            "floor", "ceil", "trunc", "sin", "cos", "tan", "exp", "log", "where",
        };

        /// <summary>Returns a minimal 1-element JSONL repro, or null if the case can't be element-wise shrunk.</summary>
        public static string ShrinkElementwise(FuzzCorpus.Case c, int diffIndex)
        {
            if (diffIndex < 0 || !ElementwiseOps.Contains(c.Op))
                return null;

            long outSize = 1;
            foreach (var d in c.Expected.Shape) outSize *= d;
            if (diffIndex >= outSize) return null;

            var minOperands = new List<string>();
            foreach (var o in c.Operands)
            {
                long osize = 1;
                foreach (var d in o.Shape) osize *= d;
                int idx = osize == 1 ? 0 : (osize == outSize ? diffIndex : -1);
                if (idx < 0) return null; // broadcast that isn't scalar/full-size: skip

                string hex = ElementHex(FuzzCorpus.Reconstruct(o), idx, o.Dtype);
                minOperands.Add(OperandJson(o.Dtype, hex));
            }

            int eisz = FuzzCorpus.DtypeToTC(c.Expected.Dtype).SizeOf();
            string expHex = Slice(c.Expected.Buffer, diffIndex, eisz);

            var paramsJson = JsonSerializer.Serialize(c.Params ?? new Dictionary<string, JsonElement>());
            return "{" +
                $"\"id\":\"shrunk/{c.Id}\",\"op\":\"{c.Op}\",\"params\":{paramsJson}," +
                $"\"operands\":[{string.Join(",", minOperands)}]," +
                $"\"expected\":{{\"dtype\":\"{c.Expected.Dtype}\",\"shape\":[],\"buffer\":\"{expHex}\"}}," +
                "\"layout\":\"shrunk\",\"valueclass\":\"shrunk\"}";
        }

        private static unsafe string ElementHex(NumSharp.NDArray nd, int idx, string dtype)
        {
            int isz = FuzzCorpus.DtypeToTC(dtype).SizeOf();
            var flat = nd.flatten();
            byte* p = (byte*)flat.Address + (long)idx * isz;
            var sb = new StringBuilder(isz * 2);
            for (int i = 0; i < isz; i++) sb.Append(p[i].ToString("x2"));
            return sb.ToString();
        }

        private static string OperandJson(string dtype, string hex)
            => $"{{\"dtype\":\"{dtype}\",\"shape\":[],\"strides\":[],\"offset\":0,\"bufferSize\":1,\"buffer\":\"{hex}\"}}";

        private static string Slice(string hex, int index, int itemSize)
            => hex.Substring(index * itemSize * 2, itemSize * 2);
    }
}
