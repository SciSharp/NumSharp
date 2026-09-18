#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// gen_decimal_oracle.cs — INDEPENDENT differential oracle for the Decimal dtype.
//
// Decimal is the ONE NumSharp numeric dtype with no NumPy analog (System.Decimal:
// 16-byte, base-10, 96-bit mantissa × 10^-scale). NumPy cannot be the oracle, so this
// generator IS the oracle: it computes every expected value with NAIVE scalar
// System.Decimal arithmetic (plain C# operators, NO NumSharp kernels), then emits the
// exact same JSONL schema as gen_oracle.py (operand = base-buffer + element shape/
// strides/offset; expected = C-contiguous result bytes). The C# harness
// (FuzzCorpus + value-aware BitDiff) replays the operand through NumSharp's decimal
// KERNELS and compares — so a divergence is a real kernel bug (strided/broadcast/
// reduction/scan iteration or accumulation), independent of the naive oracle.
//
// Run:  dotnet run gen_decimal_oracle.cs           (writes corpus/decimal_*.jsonl)
// =============================================================================
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

static class Gen
{
    // -- decimal pools: varied SCALE (1.0 vs 1.00), sign, zero, magnitude. Kept well within
    // decimal range so products don't overflow. Front-load edges (0, ±1, scale variants).
    static readonly decimal[] POOL = {
        0m, 1m, -1m, 1.0m, 1.00m, -3.5m, 2.25m, 0.1m, 0.125m, -0.125m,
        0.123456789m, 12345.6789m, -9999.99999m, 1000m, -1000m, 7m, 3m, 2m, 42m, -42m,
        0.0001m, -0.0001m, 100000m, 6m, 11m, -8m, 9.5m, -9.5m,
    };
    // Divisor pool: same spread but NEVER zero (decimal /0 throws; no NumPy oracle to mirror).
    static readonly decimal[] NZPOOL = POOL.Where(x => x != 0m).ToArray();

    static decimal[] Fill(int n, bool nonzero, int rot)
    {
        var src = nonzero ? NZPOOL : POOL;
        var a = new decimal[n];
        for (int i = 0; i < n; i++) a[i] = src[(i + rot) % src.Length];
        return a;
    }

    // ---- in-memory decimal <-> bytes (exactly how ArraySlice.FromBuffer<decimal> reads them) ----
    static string HexOf(decimal[] v)
    {
        var bytes = MemoryMarshal.AsBytes(v.AsSpan());
        var sb = new StringBuilder(bytes.Length * 2);
        foreach (var b in bytes) sb.Append(b.ToString("x2"));
        return sb.ToString();
    }
    static string HexOf(byte[] v) { var sb = new StringBuilder(v.Length * 2); foreach (var b in v) sb.Append(b.ToString("x2")); return sb.ToString(); }

    // ---- a single operand: a C-contiguous base[] + an aliasing view (shape/elem-strides/offset). ----
    sealed class Operand
    {
        public decimal[] Base;            // contiguous storage (bufferSize == Base.Length)
        public int[] Shape;
        public long[] Strides;            // element strides
        public long Offset;               // element offset
        public string Describe()          // -> operand descriptor JSON
            => $"{{\"dtype\":\"decimal\",\"shape\":[{string.Join(",", Shape)}],"
             + $"\"strides\":[{string.Join(",", Strides)}],\"offset\":{Offset},"
             + $"\"bufferSize\":{Base.Length},\"buffer\":\"{HexOf(Base)}\"}}";

        // logical values walked in C-order over Shape (offset + Σ coord*stride into Base).
        public decimal[] Logical()
        {
            int n = Shape.Aggregate(1, (a, b) => a * b);
            var outv = new decimal[n];
            if (n == 0) return outv;
            var coord = new int[Shape.Length];
            for (int i = 0; i < n; i++)
            {
                long flat = Offset;
                for (int d = 0; d < Shape.Length; d++) flat += coord[d] * Strides[d];
                outv[i] = Base[flat];
                for (int d = Shape.Length - 1; d >= 0; d--) { if (++coord[d] < Shape[d]) break; coord[d] = 0; }
            }
            return outv;
        }
    }

    // C-contiguous strides for a shape.
    static long[] CStrides(int[] shape)
    {
        var s = new long[shape.Length];
        long acc = 1;
        for (int d = shape.Length - 1; d >= 0; d--) { s[d] = acc; acc *= shape[d]; }
        return s;
    }

    // ---- single-operand layout catalog (mirrors layout_catalog.py element-strides) ----
    static Operand SingleLayout(string name, int rot, bool nonzero)
    {
        switch (name)
        {
            case "c_contiguous_1d": { var b = Fill(8, nonzero, rot); return new Operand { Base = b, Shape = new[]{8}, Strides = new long[]{1}, Offset = 0 }; }
            case "c_contiguous_2d": { var b = Fill(20, nonzero, rot); return new Operand { Base = b, Shape = new[]{4,5}, Strides = new long[]{5,1}, Offset = 0 }; }
            case "c_contiguous_3d": { var b = Fill(24, nonzero, rot); return new Operand { Base = b, Shape = new[]{2,3,4}, Strides = new long[]{12,4,1}, Offset = 0 }; }
            case "f_contiguous_2d": { var b = Fill(20, nonzero, rot); return new Operand { Base = b, Shape = new[]{4,5}, Strides = new long[]{1,4}, Offset = 0 }; }      // (5,4).T
            case "transposed_3d":   { var b = Fill(24, nonzero, rot); return new Operand { Base = b, Shape = new[]{4,2,3}, Strides = new long[]{1,12,4}, Offset = 0 }; }  // (2,3,4).transpose(2,0,1)
            case "strided_step2_1d":{ var b = Fill(16, nonzero, rot); return new Operand { Base = b, Shape = new[]{8}, Strides = new long[]{2}, Offset = 0 }; }            // [::2]
            case "negstride_1d":    { var b = Fill(8, nonzero, rot);  return new Operand { Base = b, Shape = new[]{8}, Strides = new long[]{-1}, Offset = 7 }; }           // [::-1]
            case "simple_slice_offset_1d": { var b = Fill(10, nonzero, rot); return new Operand { Base = b, Shape = new[]{5}, Strides = new long[]{1}, Offset = 2 }; }     // [2:7]
            case "strided_2d_cols": { var b = Fill(24, nonzero, rot); return new Operand { Base = b, Shape = new[]{4,3}, Strides = new long[]{6,2}, Offset = 0 }; }        // (4,6)[:,::2]
            case "broadcast_1d_to_2d": { var b = Fill(5, nonzero, rot); return new Operand { Base = b, Shape = new[]{4,5}, Strides = new long[]{0,1}, Offset = 0 }; }      // broadcast_to((4,5))
            case "scalar_0d":       { var b = Fill(1, nonzero, rot);  return new Operand { Base = b, Shape = new int[0], Strides = new long[0], Offset = 0 }; }
            case "one_element_1d":  { var b = Fill(1, nonzero, rot);  return new Operand { Base = b, Shape = new[]{1}, Strides = new long[]{1}, Offset = 0 }; }
            case "highrank_5d":     { var b = Fill(12, nonzero, rot); return new Operand { Base = b, Shape = new[]{2,1,3,1,2}, Strides = new long[]{6,6,2,2,1}, Offset = 0 }; }
            // G8b: on-demand empty (NOT in SINGLE_LAYOUTS — used only by the explicit empty cases).
            case "empty_2d":        { return new Operand { Base = new decimal[0], Shape = new[]{0,3}, Strides = new long[]{3,1}, Offset = 0 }; }
            default: throw new Exception("unknown single layout " + name);
        }
    }
    static readonly string[] SINGLE_LAYOUTS = {
        "c_contiguous_1d","c_contiguous_2d","c_contiguous_3d","f_contiguous_2d","transposed_3d",
        "strided_step2_1d","negstride_1d","simple_slice_offset_1d","strided_2d_cols",
        "broadcast_1d_to_2d","scalar_0d","one_element_1d","highrank_5d",
    };

    // ---- pairwise layout catalog (mirrors PAIR_LAYOUTS). Returns (A, B). ----
    static (Operand, Operand) PairLayout(string name, bool bNonzero)
    {
        Operand C(int[] sh, int rot, bool nz) { var b = Fill(sh.Aggregate(1,(x,y)=>x*y), nz, rot); return new Operand { Base = b, Shape = sh, Strides = CStrides(sh), Offset = 0 }; }
        switch (name)
        {
            case "pp_contig_contig": return (C(new[]{4,5},0,false), C(new[]{4,5},3,bNonzero));
            case "pp_contig_fortran": { var a = C(new[]{4,5},0,false); var bb = Fill(20,bNonzero,3); return (a, new Operand{ Base=bb, Shape=new[]{4,5}, Strides=new long[]{1,4}, Offset=0 }); }
            case "pp_strided_strided": { var ab = Fill(40,false,0); var bb = Fill(40,bNonzero,3);
                return (new Operand{Base=ab,Shape=new[]{4,5},Strides=new long[]{10,2},Offset=0}, new Operand{Base=bb,Shape=new[]{4,5},Strides=new long[]{10,2},Offset=0}); }
            case "pp_scalar_right": return (C(new[]{4,5},0,false), new Operand{ Base=Fill(1,bNonzero,3), Shape=new int[0], Strides=new long[0], Offset=0 });
            case "pp_scalar_left": return (new Operand{ Base=Fill(1,false,0), Shape=new int[0], Strides=new long[0], Offset=0 }, C(new[]{4,5},3,bNonzero));
            case "pp_broadcast_row": return (C(new[]{4,5},0,false), C(new[]{5},3,bNonzero));
            case "pp_negstride_both": { var ab=Fill(8,false,0); var bb=Fill(8,bNonzero,3);
                return (new Operand{Base=ab,Shape=new[]{8},Strides=new long[]{-1},Offset=7}, new Operand{Base=bb,Shape=new[]{8},Strides=new long[]{-1},Offset=7}); }
            // G8c: the two pair layouts the decimal tier missed (mirrors layout_catalog.py).
            case "pp_contig_strided": { var a = C(new[]{4,5},0,false); var bb = Fill(40,bNonzero,3);
                return (a, new Operand{ Base=bb, Shape=new[]{4,5}, Strides=new long[]{10,2}, Offset=0 }); }  // (4,10)[:, ::2]
            case "pp_broadcast_col": return (C(new[]{4,1},0,false), C(new[]{1,5},3,bNonzero));               // (4,1) op (1,5) -> (4,5)
            default: throw new Exception("unknown pair layout " + name);
        }
    }
    static readonly string[] PAIR_LAYOUTS = {
        "pp_contig_contig","pp_contig_fortran","pp_strided_strided","pp_scalar_right",
        "pp_scalar_left","pp_broadcast_row","pp_negstride_both",
        "pp_contig_strided","pp_broadcast_col",                       // G8c
    };

    // ---- numpy-style broadcast of two shapes ----
    static int[] BroadcastShape(int[] a, int[] b)
    {
        int n = Math.Max(a.Length, b.Length);
        var r = new int[n];
        for (int i = 0; i < n; i++)
        {
            int da = i < n - a.Length ? 1 : a[i - (n - a.Length)];
            int db = i < n - b.Length ? 1 : b[i - (n - b.Length)];
            if (da != db && da != 1 && db != 1) throw new Exception("not broadcastable");
            r[i] = Math.Max(da, db);
        }
        return r;
    }
    // value of flat-C-order `vals` (shape `shp`) at result coord (right-aligned, broadcast).
    static decimal At(decimal[] vals, int[] shp, int[] coord)
    {
        int n = coord.Length; long flat = 0; long strideAcc = 1;
        for (int d = shp.Length - 1; d >= 0; d--)
        {
            int c = coord[n - shp.Length + d];
            int idx = shp[d] == 1 ? 0 : c;
            flat += idx * strideAcc; strideAcc *= shp[d];
        }
        return vals[flat];
    }

    // ---- naive scalar oracle ops ----
    static decimal FloorDiv(decimal a, decimal b) => decimal.Floor(a / b);
    static decimal Mod(decimal a, decimal b) => a - decimal.Floor(a / b) * b;  // numpy floored remainder
    static decimal Sign(decimal a) => a > 0 ? 1m : (a < 0 ? -1m : 0m);

    static readonly (string, Func<decimal,decimal,decimal>, bool)[] BIN_ARITH = {
        ("add", (a,b)=>a+b, false), ("subtract", (a,b)=>a-b, false), ("multiply", (a,b)=>a*b, false),
        ("divide", (a,b)=>a/b, true), ("floor_divide", FloorDiv, true), ("mod", Mod, true),
        ("maximum", Math.Max, false), ("minimum", Math.Min, false),
    };
    static readonly (string, Func<decimal,decimal,bool>)[] BIN_CMP = {
        ("equal",(a,b)=>a==b), ("not_equal",(a,b)=>a!=b), ("less",(a,b)=>a<b),
        ("greater",(a,b)=>a>b), ("less_equal",(a,b)=>a<=b), ("greater_equal",(a,b)=>a>=b),
    };
    static readonly (string, Func<decimal,decimal>)[] UNARY = {
        ("negative", a=>-a), ("abs", Math.Abs), ("sign", Sign), ("square", a=>a*a),
        // rounding-toward family — exact in base-10 decimal (no NumPy analog to mirror).
        ("floor", decimal.Floor), ("ceil", decimal.Ceiling), ("trunc", decimal.Truncate),
    };

    static string Case(string id, string op, string paramsJson, IEnumerable<string> operandJsons, string expDtype, int[] expShape, string expBufHex)
        => $"{{\"id\":\"{id}\",\"op\":\"{op}\",\"params\":{paramsJson},"
         + $"\"operands\":[{string.Join(",", operandJsons)}],"
         + $"\"expected\":{{\"dtype\":\"{expDtype}\",\"shape\":[{string.Join(",", expShape)}],\"buffer\":\"{expBufHex}\"}},"
         + $"\"layout\":\"decimal\",\"valueclass\":\"decimal\"}}";

    public static void Main()
    {
        string here = AppContext.BaseDirectory;
        // resolve test/oracle dir regardless of run cwd: walk up to find NumSharp.Tests sibling.
        string oracleDir = FindOracleDir();
        string corpus = Path.GetFullPath(Path.Combine(oracleDir, "..", "NumSharp.Tests.Oracle", "Fuzz", "corpus"));
        Directory.CreateDirectory(corpus);

        var unary = new List<string>();
        var binary = new List<string>();
        var reduce = new List<string>();
        var scan = new List<string>();
        int n = 0;

        // ----- UNARY (negative/abs/sign/square) over every single layout -----
        foreach (var ln in SINGLE_LAYOUTS)
        {
            var o = SingleLayout(ln, 0, false);
            var log = o.Logical();
            foreach (var (name, f) in UNARY)
            {
                var exp = log.Select(f).ToArray();
                unary.Add(Case($"{name}/decimal/{ln}/{n++}", name, "{}", new[]{o.Describe()}, "decimal", o.Shape, HexOf(exp)));
            }
        }

        // ----- BINARY arith + comparison over pair layouts -----
        foreach (var ln in PAIR_LAYOUTS)
        {
            foreach (var (name, f, nz) in BIN_ARITH)
            {
                var (a, b) = PairLayout(ln, nz);
                var la = a.Logical(); var lb = b.Logical();
                var rs = BroadcastShape(a.Shape, b.Shape);
                var exp = BroadcastApply(la, a.Shape, lb, b.Shape, rs, f);
                binary.Add(Case($"{name}/decimal/{ln}/{n++}", name, "{}", new[]{a.Describe(), b.Describe()}, "decimal", rs, HexOf(exp)));
            }
            foreach (var (name, f) in BIN_CMP)
            {
                var (a, b) = PairLayout(ln, false);
                var la = a.Logical(); var lb = b.Logical();
                var rs = BroadcastShape(a.Shape, b.Shape);
                var exp = BroadcastApplyBool(la, a.Shape, lb, b.Shape, rs, f);
                binary.Add(Case($"{name}/decimal/{ln}/{n++}", name, "{}", new[]{a.Describe(), b.Describe()}, "bool", rs, HexOf(exp)));
            }
        }

        // ----- REDUCTIONS (axis=None): sum/prod/min/max/mean -----
        foreach (var ln in SINGLE_LAYOUTS)
        {
            if (ln == "scalar_0d") continue; // reduce of 0-D is identity; skip the degenerate
            var o = SingleLayout(ln, 1, false);
            var log = o.Logical();
            if (log.Length == 0) continue;
            void Add(string op, decimal val) => reduce.Add(Case($"{op}/decimal/{ln}/{n++}", op, "{}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{val})));
            Add("sum", log.Aggregate(0m, (x,y)=>x+y));
            Add("prod", ProdSafe(log));
            Add("min", log.Min());
            Add("max", log.Max());
            Add("mean", log.Aggregate(0m,(x,y)=>x+y) / log.Length);
        }

        // ----- G8a: AXIS reductions sum/min/max/mean × axis {0,last} × keepdims {F,T} over the
        // three 2-D layouts. decimal + is exact so accumulation order is irrelevant; mean is one
        // division of the same exact sum on both sides. -----
        foreach (var ln in new[] { "c_contiguous_2d", "f_contiguous_2d", "strided_2d_cols" })
        {
            var o = SingleLayout(ln, 1, false);
            var log = o.Logical();                       // C-order logical values
            int rows = o.Shape[0], cols = o.Shape[1];
            foreach (int axis in new[] { 0, 1 })
            {
                int outN = axis == 0 ? cols : rows;
                int m = axis == 0 ? rows : cols;
                var sums = new decimal[outN]; var mins = new decimal[outN];
                var maxs = new decimal[outN]; var means = new decimal[outN];
                for (int j = 0; j < outN; j++)
                {
                    decimal acc = 0m, mn = 0m, mx = 0m;
                    for (int i = 0; i < m; i++)
                    {
                        decimal v = axis == 0 ? log[i * cols + j] : log[j * cols + i];
                        acc += v;
                        if (i == 0) { mn = v; mx = v; }
                        else { if (v < mn) mn = v; if (v > mx) mx = v; }
                    }
                    sums[j] = acc; mins[j] = mn; maxs[j] = mx; means[j] = acc / m;
                }
                foreach (var kd in new[] { false, true })
                {
                    int[] shp = kd ? (axis == 0 ? new[] { 1, cols } : new[] { rows, 1 }) : new[] { outN };
                    string pj = $"{{\"axis\":{axis},\"keepdims\":{(kd ? "true" : "false")}}}";
                    void AddAx(string op, decimal[] vals) => reduce.Add(Case(
                        $"{op}/decimal/{ln}/ax{axis}kd{(kd ? 1 : 0)}/{n++}", op, pj,
                        new[] { o.Describe() }, "decimal", shp, HexOf(vals)));
                    AddAx("sum", sums); AddAx("min", mins); AddAx("max", maxs); AddAx("mean", means);
                }
            }
        }

        // ----- G8b: EMPTY decimal — sum(empty)=0m, prod(empty)=1m (flat; NumPy identity values). -----
        {
            var o = SingleLayout("empty_2d", 0, false);
            reduce.Add(Case($"sum/decimal/empty_2d/{n++}", "sum", "{}", new[] { o.Describe() },
                "decimal", new int[0], HexOf(new[] { 0m })));
            reduce.Add(Case($"prod/decimal/empty_2d/{n++}", "prod", "{}", new[] { o.Describe() },
                "decimal", new int[0], HexOf(new[] { 1m })));
        }

        // ----- G8e: flat argmax/argmin (-> int64, first-occurrence on value ties), all/any
        // (-> bool, x != 0m truthiness), count_nonzero (-> int64). argmax/argmin/count_nonzero
        // replay with axis=0 over 1-D layouts (== flatten for 1-D; the harness has no
        // flatten-argmax overload); all/any replay flat (axis=None). -----
        foreach (var ln in new[] { "c_contiguous_1d", "strided_step2_1d", "negstride_1d", "simple_slice_offset_1d" })
        {
            var o = SingleLayout(ln, 1, false);
            var log = o.Logical();
            int iMax = 0, iMin = 0; long nz = 0;
            bool all = true, any = false;
            for (int i = 0; i < log.Length; i++)
            {
                if (log[i] > log[iMax]) iMax = i;
                if (log[i] < log[iMin]) iMin = i;
                if (log[i] != 0m) { nz++; any = true; } else all = false;
            }
            string pj0 = "{\"axis\":0,\"keepdims\":false}";
            reduce.Add(Case($"argmax/decimal/{ln}/{n++}", "argmax", pj0, new[] { o.Describe() },
                "int64", new int[0], HexOf(Bytes(new[] { (long)iMax }))));
            reduce.Add(Case($"argmin/decimal/{ln}/{n++}", "argmin", pj0, new[] { o.Describe() },
                "int64", new int[0], HexOf(Bytes(new[] { (long)iMin }))));
            // G13: the FLAT (axis=None) form takes argmax_elementwise_il — a DIFFERENT code path
            // from the axis kernel above, and the one whose Decimal IL compare was silently wrong
            // (argmax([3,9,1,5]) returned 0 while this tier stayed green through the axis path).
            string pjFlat = "{\"axis\":null,\"keepdims\":false}";
            reduce.Add(Case($"argmax_flat/decimal/{ln}/{n++}", "argmax", pjFlat, new[] { o.Describe() },
                "int64", new int[0], HexOf(Bytes(new[] { (long)iMax }))));
            reduce.Add(Case($"argmin_flat/decimal/{ln}/{n++}", "argmin", pjFlat, new[] { o.Describe() },
                "int64", new int[0], HexOf(Bytes(new[] { (long)iMin }))));
            reduce.Add(Case($"count_nonzero/decimal/{ln}/{n++}", "count_nonzero", pj0, new[] { o.Describe() },
                "int64", new int[0], HexOf(Bytes(new[] { nz }))));
            reduce.Add(Case($"all/decimal/{ln}/{n++}", "all", "{}", new[] { o.Describe() },
                "bool", new int[0], HexOf(new[] { (byte)(all ? 1 : 0) })));
            reduce.Add(Case($"any/decimal/{ln}/{n++}", "any", "{}", new[] { o.Describe() },
                "bool", new int[0], HexOf(new[] { (byte)(any ? 1 : 0) })));
        }

        // ----- SCAN (axis=None -> flatten): cumsum/cumprod -----
        foreach (var ln in SINGLE_LAYOUTS)
        {
            if (ln == "scalar_0d") continue;
            var o = SingleLayout(ln, 2, false);
            var log = o.Logical();
            if (log.Length == 0) continue;
            var cs = new decimal[log.Length]; var cp = new decimal[log.Length];
            decimal s = 0m, p = 1m;
            for (int i = 0; i < log.Length; i++) { s += log[i]; cs[i] = s; p *= log[i]; cp[i] = p; }
            scan.Add(Case($"cumsum/decimal/{ln}/{n++}", "cumsum", "{\"axis\":null}", new[]{o.Describe()}, "decimal", new[]{log.Length}, HexOf(cs)));
            scan.Add(Case($"cumprod/decimal/{ln}/{n++}", "cumprod", "{\"axis\":null}", new[]{o.Describe()}, "decimal", new[]{log.Length}, HexOf(cp)));

            // diff n=1,2 along the LAST axis (a[1:]-a[:-1]); output shrinks by n on that axis.
            if (o.Shape.Length >= 1)
            {
                int last = o.Shape[o.Shape.Length - 1];
                foreach (int nd in new[] { 1, 2 })
                {
                    if (last < nd + 1) continue;                       // need ≥2 for n=1, ≥3 for n=2
                    var dres = DiffAxis(log, o.Shape, nd, -1);
                    var dsh = (int[])o.Shape.Clone(); dsh[dsh.Length - 1] -= nd;
                    scan.Add(Case($"diff/decimal/{ln}/n{nd}/{n++}", "diff", $"{{\"n\":{nd},\"axis\":-1}}", new[]{o.Describe()}, "decimal", dsh, HexOf(dres)));
                }
            }
        }

        var power = new List<string>();
        var varstd = new List<string>();
        var matmul = new List<string>();
        var astype = new List<string>();
        var stat = new List<string>();
        var where = new List<string>();
        var sort = new List<string>();
        var manip = new List<string>();
        // G14 (2026-09-18): the coverage-audit expansion. These are the decimal-capable ops the tier
        // MISSED — and decimal is the ONLY dtype-scope this whole differential pipeline exercises for
        // them (no other corpus file carries a decimal operand), so an unexercised branch here is a
        // live silent-bug risk (the class that hid the G13 flat-argmax 16-byte IL-compare bug).
        var extra = new List<string>();       // reciprocal/positive/fabs/rint/round_/signbit/spacing/deg2rad/rad2deg/modf
        var transcend = new List<string>();   // sqrt/cbrt/exp/log/trig/... via the exact decimal->double->Math.*->decimal bridge (HOST-PINNED)
        var select = new List<string>();      // take/put/place/putmask/select/choose/compress/extract/take_along_axis (16-byte gather/scatter)
        var products = new List<string>();    // dot/inner/outer/vdot/tensordot/trace/kron (scalar decimal MAC — matmul was the ONLY covered product)
        var search = new List<string>();      // argsort/searchsorted/unique/nonzero/flatnonzero/lexsort (decimal compare-driven sort/search)

        // ----- POWER decimal^int (exact: repeated multiply / reciprocal). Exponent is a 0-D
        // decimal whose value is a whole number — DecimalMath.Pow must be exact for integer powers.
        // G7 (F7): negative exponents were dead code (loop was {0,1,2,3} while IntPow and the
        // nonzero-base plumbing below already supported e<0). decimal^-n = 1/(a^n), exact oracle. -----
        foreach (var ln in new[] { "c_contiguous_1d", "c_contiguous_2d", "strided_step2_1d", "negstride_1d", "broadcast_1d_to_2d" })
        {
            foreach (int e in new[] { -2, -1, 0, 1, 2, 3 })
            {
                var a = SingleLayout(ln, 0, e < 0);                 // nonzero base only when exponent<0
                var b = new Operand { Base = new[] { (decimal)e }, Shape = new int[0], Strides = new long[0], Offset = 0 };
                var la = a.Logical();
                var exp = la.Select(x => IntPow(x, e)).ToArray();
                power.Add(Case($"power/decimal/{ln}^{e}/{n++}", "power", "{}", new[]{a.Describe(), b.Describe()}, "decimal", a.Shape, HexOf(exp)));
            }
        }

        // ----- VAR / STD (axis=None, ddof=0). var = mean((x-mean)^2) is EXACT decimal arithmetic
        // (no sqrt). std = sqrt(var) uses an INDEPENDENT Newton decimal sqrt as the oracle. -----
        foreach (var ln in SINGLE_LAYOUTS)
        {
            if (ln == "scalar_0d" || ln == "one_element_1d") continue; // var of 1 elem = 0 (degenerate)
            var o = SingleLayout(ln, 4, false);
            var log = o.Logical();
            if (log.Length == 0) continue;
            decimal mean = log.Aggregate(0m,(x,y)=>x+y) / log.Length;
            decimal v = log.Aggregate(0m,(acc,x)=>acc + (x-mean)*(x-mean)) / log.Length;
            varstd.Add(Case($"var/decimal/{ln}/{n++}", "var", "{}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{v})));
            varstd.Add(Case($"std/decimal/{ln}/{n++}", "std", "{}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{DecSqrt(v)})));
        }

        // ----- MATMUL 2D@2D (exact: decimal + is exact, so accumulation order is irrelevant). -----
        foreach (var (m, k, p2, fortranB) in new[] { (3,4,2,false), (2,3,3,false), (4,2,5,true), (1,4,1,false) })
        {
            var A = new Operand { Base = Fill(m*k, false, 0), Shape = new[]{m,k}, Strides = CStrides(new[]{m,k}), Offset = 0 };
            Operand B = fortranB
                ? new Operand { Base = Fill(k*p2, false, 5), Shape = new[]{k,p2}, Strides = new long[]{1,k}, Offset = 0 } // (p2,k).T
                : new Operand { Base = Fill(k*p2, false, 5), Shape = new[]{k,p2}, Strides = CStrides(new[]{k,p2}), Offset = 0 };
            var la = A.Logical(); var lb = B.Logical();
            var exp = new decimal[m*p2];
            for (int i = 0; i < m; i++) for (int j = 0; j < p2; j++) { decimal acc = 0m; for (int t = 0; t < k; t++) acc += la[i*k+t]*lb[t*p2+j]; exp[i*p2+j] = acc; }
            matmul.Add(Case($"matmul/decimal/{m}x{k}@{k}x{p2}{(fortranB?"F":"")}/{n++}", "matmul", "{}", new[]{A.Describe(), B.Describe()}, "decimal", new[]{m,p2}, HexOf(exp)));
        }

        // ----- ASTYPE decimal->X and X->decimal (the cast kernel). Values kept in-range. -----
        // decimal -> wider numeric (no overflow for the small-value pool).
        foreach (var ln in new[] { "c_contiguous_1d", "c_contiguous_2d", "strided_step2_1d", "negstride_1d" })
        {
            var o = SmallSingle(ln);
            var log = o.Logical();
            void Dto(string dt, string dtype, byte[] buf) => astype.Add(Case($"astype/decimal->{dt}/{ln}/{n++}", "astype", $"{{\"dtype\":\"{dtype}\"}}", new[]{o.Describe()}, dtype, o.Shape, HexOf(buf)));
            Dto("int64", "int64", Bytes(log.Select(x => (long)x).ToArray()));
            Dto("int32", "int32", Bytes(log.Select(x => (int)x).ToArray()));
            Dto("float64", "float64", Bytes(log.Select(x => (double)x).ToArray()));
            Dto("float32", "float32", Bytes(log.Select(x => (float)x).ToArray()));
        }
        // X -> decimal (always representable).
        foreach (var ln in new[] { "c_contiguous_1d", "c_contiguous_2d" })
        {
            int n0 = ln == "c_contiguous_1d" ? 6 : 12; int[] sh = ln == "c_contiguous_1d" ? new[]{6} : new[]{3,4};
            long[] iv = Enumerable.Range(0, n0).Select(i => (long)(i % 2 == 0 ? i : -i)).ToArray();
            double[] dv = Enumerable.Range(0, n0).Select(i => (i - 2) * 1.25).ToArray();
            astype.Add(AstypeTo("int64", sh, Bytes(iv), HexOf(iv.Select(x => (decimal)x).ToArray()), ref n));
            astype.Add(AstypeTo("int32", sh, Bytes(iv.Select(x=>(int)x).ToArray()), HexOf(iv.Select(x => (decimal)x).ToArray()), ref n));
            astype.Add(AstypeTo("float64", sh, Bytes(dv), HexOf(dv.Select(x => (decimal)x).ToArray()), ref n));
        }

        // ----- G8d: astype decimal <-> {bool, int16, uint64} (exact for the pools used). -----
        foreach (var ln in new[] { "c_contiguous_1d", "strided_step2_1d" })
        {
            var o = SmallSingle(ln);                        // pool has 0m -> real bool mix
            var log = o.Logical();
            astype.Add(Case($"astype/decimal->bool/{ln}/{n++}", "astype", "{\"dtype\":\"bool\"}",
                new[] { o.Describe() }, "bool", o.Shape,
                HexOf(log.Select(x => (byte)(x != 0m ? 1 : 0)).ToArray())));
            astype.Add(Case($"astype/decimal->int16/{ln}/{n++}", "astype", "{\"dtype\":\"int16\"}",
                new[] { o.Describe() }, "int16", o.Shape,
                HexOf(Bytes(log.Select(x => (short)x).ToArray()))));            // truncation toward zero
        }
        {
            // decimal -> uint64: NONNEGATIVE pool only (C# decimal->ulong throws for negatives;
            // NumPy has no oracle to mirror a modular wrap for base-10, so stay in-range).
            decimal[] np2 = { 0m, 1m, 2.75m, 7.9m, 42.5m, 100m, 9.99m, 3m };
            var o = new Operand { Base = np2, Shape = new[]{8}, Strides = new long[]{1}, Offset = 0 };
            astype.Add(Case($"astype/decimal->uint64/nonneg/{n++}", "astype", "{\"dtype\":\"uint64\"}",
                new[] { o.Describe() }, "uint64", o.Shape,
                HexOf(Bytes(np2.Select(x => (ulong)x).ToArray()))));
        }
        {
            // {bool, int16, uint64} -> decimal (always exact).
            byte[] bv = { 1, 0, 1, 1, 0, 0 };
            astype.Add(AstypeTo("bool", new[]{6}, bv, HexOf(bv.Select(x => (decimal)x).ToArray()), ref n));
            short[] sv = { 0, -1, 127, -128, 32767, -32768 };
            astype.Add(AstypeTo("int16", new[]{6}, Bytes(sv), HexOf(sv.Select(x => (decimal)x).ToArray()), ref n));
            ulong[] uv = { 0, 1, 255, 65536, 4294967295, 9000000000000000000 };
            astype.Add(AstypeTo("uint64", new[]{6}, Bytes(uv), HexOf(uv.Select(x => (decimal)x).ToArray()), ref n));
        }

        // ----- STAT (axis=None -> scalar): clip / median / ptp / percentile / quantile.
        // clip is elementwise (Max(lo,Min(hi,x))); the order stats flatten the logical view. -----
        foreach (var ln in SINGLE_LAYOUTS)
        {
            var o = SingleLayout(ln, 1, false);
            var log = o.Logical();
            if (log.Length == 0) continue;

            // clip against two scalar (lo,hi) windows — 0-D decimal bound operands.
            foreach (var (lo, hi) in new[] { (-5m, 10m), (0m, 100m) })
            {
                var loOp = new Operand { Base = new[]{ lo }, Shape = new int[0], Strides = new long[0], Offset = 0 };
                var hiOp = new Operand { Base = new[]{ hi }, Shape = new int[0], Strides = new long[0], Offset = 0 };
                var cexp = log.Select(x => Math.Max(lo, Math.Min(hi, x))).ToArray();
                stat.Add(Case($"clip/decimal/{ln}/[{lo},{hi}]/{n++}", "clip", "{}", new[]{o.Describe(), loOp.Describe(), hiOp.Describe()}, "decimal", o.Shape, HexOf(cexp)));
            }

            if (ln == "scalar_0d") continue; // order stats over 0-D are identity; skip the degenerate
            var flat = (decimal[])log.Clone(); Array.Sort(flat);
            stat.Add(Case($"median/decimal/{ln}/{n++}", "median", "{}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{ Median(flat) })));
            stat.Add(Case($"ptp/decimal/{ln}/{n++}", "ptp", "{}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{ flat[flat.Length-1] - flat[0] })));
            // percentile q in {0,25,50,75,100}; quantile q in {0,.25,.5,.75,1} — same order statistic.
            foreach (var (pq, frac) in new[] { ("0.0", 0.0), ("25.0", 0.25), ("50.0", 0.5), ("75.0", 0.75), ("100.0", 1.0) })
                stat.Add(Case($"percentile/decimal/{ln}/p{pq}/{n++}", "percentile", $"{{\"q\":{pq}}}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{ Quantile(flat, frac) })));
            foreach (var (qq, frac) in new[] { ("0.0", 0.0), ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75), ("1.0", 1.0) })
                stat.Add(Case($"quantile/decimal/{ln}/q{qq}/{n++}", "quantile", $"{{\"q\":{qq}}}", new[]{o.Describe()}, "decimal", new int[0], HexOf(new[]{ Quantile(flat, frac) })));
        }

        // ----- WHERE (cond ? a : b): cond is a bool mask, a/b are decimal. Exercises the 16-byte
        // conditional-copy kernel over contiguous AND strided decimal operands. -----
        foreach (var (ln, aRot, bRot) in new[] { ("c_contiguous_1d", 0, 4), ("c_contiguous_2d", 0, 7), ("strided_step2_1d", 0, 4), ("negstride_1d", 0, 3) })
        {
            var a = SingleLayout(ln, aRot, false);
            // b: a fresh CONTIGUOUS decimal of the same shape (different values via rotation).
            var bBase = Fill(a.Shape.Aggregate(1,(x,y)=>x*y), false, bRot);
            var b = new Operand { Base = bBase, Shape = a.Shape, Strides = CStrides(a.Shape), Offset = 0 };
            var la = a.Logical(); var lb = b.Logical();
            var mask = la.Select(x => (byte)(x > 0m ? 1 : 0)).ToArray();   // cond = a>0
            var wexp = new decimal[la.Length];
            for (int i = 0; i < la.Length; i++) wexp[i] = la[i] > 0m ? la[i] : lb[i];
            where.Add(Case($"where/decimal/{ln}/{n++}", "where", "{}", new[]{ BoolOperandDesc(mask, a.Shape), a.Describe(), b.Describe() }, "decimal", a.Shape, HexOf(wexp)));
        }

        // ----- SORT along an axis (1-D and 2-D, contiguous + strided) -----
        foreach (var (ln, axes) in new (string, int[])[] {
            ("c_contiguous_1d", new[]{0}), ("c_contiguous_2d", new[]{0,1}),
            ("strided_step2_1d", new[]{0}), ("negstride_1d", new[]{0}), ("strided_2d_cols", new[]{0,1}) })
        {
            var o = SingleLayout(ln, 3, false);
            var log = o.Logical();
            foreach (int ax in axes)
            {
                var sres = SortAxis(log, o.Shape, ax);
                sort.Add(Case($"sort/decimal/{ln}/ax{ax}/{n++}", "sort", $"{{\"axis\":{ax}}}", new[]{o.Describe()}, "decimal", o.Shape, HexOf(sres)));
            }
        }

        // ----- MANIP (value-preserving reindex): ravel / transpose / reshape. Forces the strided
        // decimal materialize/copy path (result is compared C-contiguous via ascontiguousarray). -----
        foreach (var ln in SINGLE_LAYOUTS)
        {
            var o = SingleLayout(ln, 2, false);
            var log = o.Logical();
            int cnt = log.Length;
            // ravel -> flat C-order
            manip.Add(Case($"ravel/decimal/{ln}/{n++}", "ravel", "{}", new[]{o.Describe()}, "decimal", new[]{cnt}, HexOf(log)));
            // transpose -> reversed axes
            manip.Add(Case($"transpose/decimal/{ln}/{n++}", "transpose", "{}", new[]{o.Describe()}, "decimal", RevShape(o.Shape), HexOf(TransposeReverse(log, o.Shape))));
            // reshape -> a distinct 2-D factorization (skip when trivial: <2-D result or prime)
            if (cnt >= 2)
            {
                int rows = LargestFactorLeqSqrt(cnt);
                if (rows > 1) { var rsh = new[]{ rows, cnt / rows };
                    manip.Add(Case($"reshape/decimal/{ln}/[{rows},{cnt/rows}]/{n++}", "reshape", $"{{\"shape\":[{rows},{cnt/rows}]}}", new[]{o.Describe()}, "decimal", rsh, HexOf(log))); }
            }
        }

        // =====================================================================================
        // G14 (2026-09-18) COVERAGE-AUDIT EXPANSION. Everything below closes a decimal gap the tier
        // missed. decimal is the ONLY dtype scope the whole differential pipeline exercises for these
        // ops (no other corpus file carries a decimal operand), and it is the widest (16-byte),
        // scalar-only lane — the branch class that hid the G13 flat-argmax IL-compare bug.
        // =====================================================================================

        // ----- EXTRA: extended unary decimal->decimal (reciprocal/positive/fabs/rint/spacing/
        // deg2rad/rad2deg + modf split), signbit (->bool), and round_ at decimals {-1,0,1,2}
        // (the PyArray_Round path, DISTINCT from rint's UnaryOp.Round). All exact or portable
        // (spacing/deg2rad/rad2deg bridge through double but are pure arithmetic, no libm) -> strict tier. -----
        var EXTRA_UNARY = new (string, Func<decimal, decimal>)[]
        {
            ("positive", x => x),
            ("fabs",     Math.Abs),
            ("rint",     x => Math.Round(x, MidpointRounding.ToEven)),
            ("spacing",  SpacingDec),
            ("deg2rad",  x => (decimal)((double)x * (Math.PI / 180.0))),
            ("rad2deg",  x => (decimal)((double)x * (180.0 / Math.PI))),
            ("modf_frac", x => x - decimal.Truncate(x)),
            ("modf_int",  decimal.Truncate),
        };
        foreach (var ln in new[] { "c_contiguous_1d", "c_contiguous_2d", "strided_step2_1d", "negstride_1d" })
        {
            var o = SingleLayout(ln, 5, false);
            var log = o.Logical();
            foreach (var (name, f) in EXTRA_UNARY)
                extra.Add(Case($"{name}/decimal/{ln}/{n++}", name, "{}", new[] { o.Describe() },
                    "decimal", o.Shape, HexOf(log.Select(f).ToArray())));
            // signbit -> bool (strictly-negative test; -0m reads non-negative, ties 0 under BitDiff).
            extra.Add(Case($"signbit/decimal/{ln}/{n++}", "signbit", "{}", new[] { o.Describe() },
                "bool", o.Shape, HexOf(log.Select(x => (byte)(Math.Sign(x) < 0 ? 1 : 0)).ToArray())));
            // reciprocal 1/x -> needs a nonzero source (decimal 1/0 throws on both sides).
            var onz = SingleLayout(ln, 5, true);
            var lognz = onz.Logical();
            extra.Add(Case($"reciprocal/decimal/{ln}/{n++}", "reciprocal", "{}", new[] { onz.Describe() },
                "decimal", onz.Shape, HexOf(lognz.Select(x => 1m / x).ToArray())));
            // round_ at several decimals — banker's; base-10 makes scale/round/unscale exact.
            foreach (int dec in new[] { -1, 0, 1, 2 })
                extra.Add(Case($"round_/decimal/{ln}/d{dec}/{n++}", "round_", $"{{\"decimals\":{dec}}}",
                    new[] { o.Describe() }, "decimal", o.Shape, HexOf(log.Select(x => RoundDec(x, dec)).ToArray())));
        }

        // ----- TRANSCEND: sqrt/cbrt/exp/exp2/expm1/log/log2/log10/log1p/sin/cos/tan/sinh/cosh/tanh/
        // arcsin/arccos/arctan/arcsinh/arccosh/arctanh. The kernel computes each as the EXACT
        // (decimal)Math.X((double)v) bridge (DirectILKernelGenerator.Unary.Decimal.cs), so the oracle
        // replicates that bridge per LOGICAL element — a divergence is a decimal iteration bug, not a
        // math difference. HOST-PINNED (RunHostLibmCorpus): Math.Exp/Log/Sin/... are win-amd64 CRT libm,
        // so the cast-to-decimal bytes reproduce only on the authoring host (Inconclusive off-Windows).
        // Inputs are DOMAIN-SAFE: `(decimal)NaN/Inf` throws, so log/sqrt take >0, asin/acos [-1,1], etc. -----
        {
            decimal[] POS = { 0.1m, 0.5m, 1m, 2m, 2.5m, 4m, 7m, 10m };          // > 0     (sqrt/log/log2/log10)
            decimal[] ANY = { -8m, -2m, -0.5m, 0m, 0.5m, 2m, 3.375m, 8m };      // any     (cbrt)
            decimal[] SMALL = { -3m, -1.5m, -0.5m, 0m, 0.5m, 1m, 2m, 3m };      // small   (exp/trig/sinh/arctan/arcsinh)
            decimal[] UNIT = { -0.9m, -0.5m, -0.25m, 0m, 0.25m, 0.5m, 0.75m, 0.9m }; // (-1,1) (asin/acos/atanh)
            decimal[] GTM1 = { -0.9m, -0.5m, 0m, 0.5m, 1m, 2m, 5m, 9m };        // > -1    (log1p)
            decimal[] GE1 = { 1m, 1.25m, 2m, 3m, 5m, 8m, 1.5m, 4m };           // >= 1    (acosh)
            var TRANSCEND = new (string, decimal[], Func<decimal, decimal>)[]
            {
                ("sqrt",     POS,   x => (decimal)Math.Sqrt((double)x)),
                ("cbrt",     ANY,   x => (decimal)Math.Cbrt((double)x)),
                ("exp",      SMALL, x => (decimal)Math.Exp((double)x)),
                ("exp2",     SMALL, x => (decimal)Math.Pow(2.0, (double)x)),
                ("expm1",    SMALL, x => (decimal)(Math.Exp((double)x) - 1.0)),
                ("log",      POS,   x => (decimal)Math.Log((double)x)),
                ("log2",     POS,   x => (decimal)Math.Log2((double)x)),
                ("log10",    POS,   x => (decimal)Math.Log10((double)x)),
                ("log1p",    GTM1,  x => (decimal)Math.Log(1.0 + (double)x)),
                ("sin",      SMALL, x => (decimal)Math.Sin((double)x)),
                ("cos",      SMALL, x => (decimal)Math.Cos((double)x)),
                ("tan",      SMALL, x => (decimal)Math.Tan((double)x)),
                ("sinh",     SMALL, x => (decimal)Math.Sinh((double)x)),
                ("cosh",     SMALL, x => (decimal)Math.Cosh((double)x)),
                ("tanh",     SMALL, x => (decimal)Math.Tanh((double)x)),
                ("arcsin",   UNIT,  x => (decimal)Math.Asin((double)x)),
                ("arccos",   UNIT,  x => (decimal)Math.Acos((double)x)),
                ("arctan",   SMALL, x => (decimal)Math.Atan((double)x)),
                ("arcsinh",  SMALL, x => (decimal)Math.Asinh((double)x)),
                ("arccosh",  GE1,   x => (decimal)Math.Acosh((double)x)),
                ("arctanh",  UNIT,  x => (decimal)Math.Atanh((double)x)),
            };
            // Build a 1-D operand whose Logical()==vals in each of three layouts (contiguous / step-2 /
            // reversed) so the transcendental exercises the strided + negative-stride iteration.
            Operand TOp(decimal[] vals, string ln)
            {
                switch (ln)
                {
                    case "c_contiguous_1d":
                        return new Operand { Base = (decimal[])vals.Clone(), Shape = new[] { vals.Length }, Strides = new long[] { 1 }, Offset = 0 };
                    case "strided_step2_1d":
                        { var b = new decimal[vals.Length * 2]; for (int i = 0; i < vals.Length; i++) b[2 * i] = vals[i]; return new Operand { Base = b, Shape = new[] { vals.Length }, Strides = new long[] { 2 }, Offset = 0 }; }
                    case "negstride_1d":
                        { var b = (decimal[])vals.Clone(); Array.Reverse(b); return new Operand { Base = b, Shape = new[] { vals.Length }, Strides = new long[] { -1 }, Offset = vals.Length - 1 }; }
                    default: throw new Exception("TOp layout " + ln);
                }
            }
            foreach (var (name, pool, f) in TRANSCEND)
                foreach (var ln in new[] { "c_contiguous_1d", "strided_step2_1d", "negstride_1d" })
                {
                    var o = TOp(pool, ln);
                    transcend.Add(Case($"{name}/decimal/{ln}/{n++}", name, "{}", new[] { o.Describe() },
                        "decimal", o.Shape, HexOf(pool.Select(f).ToArray())));
                }
        }

        // ----- BINARY2: fmod (truncated remainder, sign of DIVIDEND — distinct from the covered floored
        // mod), heaviside, copysign — over the pair layouts. Appended to the binary corpus. -----
        foreach (var ln in PAIR_LAYOUTS)
        {
            var (af, bf) = PairLayout(ln, true);   // fmod: nonzero divisor
            var rsf = BroadcastShape(af.Shape, bf.Shape);
            binary.Add(Case($"fmod/decimal/{ln}/{n++}", "fmod", "{}", new[] { af.Describe(), bf.Describe() },
                "decimal", rsf, HexOf(BroadcastApply(af.Logical(), af.Shape, bf.Logical(), bf.Shape, rsf, Fmod))));
            var (a2, b2) = PairLayout(ln, false);
            var rs2 = BroadcastShape(a2.Shape, b2.Shape);
            binary.Add(Case($"heaviside/decimal/{ln}/{n++}", "heaviside", "{}", new[] { a2.Describe(), b2.Describe() },
                "decimal", rs2, HexOf(BroadcastApply(a2.Logical(), a2.Shape, b2.Logical(), b2.Shape, rs2, Heaviside))));
            binary.Add(Case($"copysign/decimal/{ln}/{n++}", "copysign", "{}", new[] { a2.Describe(), b2.Describe() },
                "decimal", rs2, HexOf(BroadcastApply(a2.Logical(), a2.Shape, b2.Logical(), b2.Shape, rs2, Copysign))));
        }

        // ----- SELECT: the 16-byte gather/scatter/conditional-copy family. Only `where` was covered.
        // Each expected value is a pure reindex/filter of the decimal logical values. -----
        {
            decimal[] sv  = { 1m, -2m, 3m, -4m, 5m, 6m, -7m, 8m };
            decimal[] sv2 = { 10m, 20m, 30m, 40m, 50m, 60m, 70m, 80m };
            var svShape = new[] { 8 };
            Operand SvC() => new Operand { Base = (decimal[])sv.Clone(), Shape = svShape, Strides = new long[] { 1 }, Offset = 0 };
            // take (axis=0) over a contiguous AND a reversed source (16-byte strided gather).
            int[] tIdx = { 0, 2, 4, 7, 1 };
            var tExp = tIdx.Select(k => sv[k]).ToArray();
            select.Add(Case($"take/decimal/c/{n++}", "take", "{\"axis\":0}",
                new[] { SvC().Describe(), Int32OperandDesc(tIdx, new[] { tIdx.Length }) },
                "decimal", new[] { tIdx.Length }, HexOf(tExp)));
            { var neg = new Operand { Base = ((Func<decimal[]>)(() => { var b = (decimal[])sv.Clone(); Array.Reverse(b); return b; }))(),
                                      Shape = svShape, Strides = new long[] { -1 }, Offset = 7 };  // logical == sv
              select.Add(Case($"take/decimal/negstride/{n++}", "take", "{\"axis\":0}",
                new[] { neg.Describe(), Int32OperandDesc(tIdx, new[] { tIdx.Length }) },
                "decimal", new[] { tIdx.Length }, HexOf(tExp))); }
            // take_along_axis (axis=0, int64 indices).
            long[] taIdx = { 3, 1, 0, 2, 6, 7 };
            select.Add(Case($"take_along_axis/decimal/c/{n++}", "take_along_axis", "{\"axis\":0}",
                new[] { SvC().Describe(), Int64OperandDesc(taIdx, new[] { taIdx.Length }) },
                "decimal", new[] { taIdx.Length }, HexOf(taIdx.Select(k => sv[(int)k]).ToArray())));
            // put (mutating): sv with selected positions overwritten. values length == indices length.
            { int[] pIdx = { 0, 2, 5 }; decimal[] pVal = { 90m, 91m, 92m };
              var exp = (decimal[])sv.Clone(); for (int i = 0; i < pIdx.Length; i++) exp[pIdx[i]] = pVal[i];
              select.Add(Case($"put/decimal/c/{n++}", "put", "{}",
                new[] { SvC().Describe(), Int32OperandDesc(pIdx, new[] { pIdx.Length }),
                        new Operand { Base = pVal, Shape = new[] { pVal.Length }, Strides = new long[] { 1 }, Offset = 0 }.Describe() },
                "decimal", svShape, HexOf(exp))); }
            // place (mutating): a.flat[mask] = values CYCLED per True. mask = sv > 0.
            { var mask = sv.Select(x => (byte)(x > 0m ? 1 : 0)).ToArray(); decimal[] plVal = { 70m, 71m };
              var exp = (decimal[])sv.Clone(); int j = 0;
              for (int i = 0; i < exp.Length; i++) if (mask[i] != 0) exp[i] = plVal[(j++) % plVal.Length];
              select.Add(Case($"place/decimal/c/{n++}", "place", "{}",
                new[] { SvC().Describe(), BoolOperandDesc(mask, svShape),
                        new Operand { Base = plVal, Shape = new[] { plVal.Length }, Strides = new long[] { 1 }, Offset = 0 }.Describe() },
                "decimal", svShape, HexOf(exp))); }
            // putmask (mutating): a.flat[i] = values[i % nv] WHERE mask (position cursor, not per-True).
            { var mask = sv.Select(x => (byte)(x > 0m ? 1 : 0)).ToArray(); decimal[] pmVal = { 70m, 71m, 72m };
              var exp = (decimal[])sv.Clone();
              for (int i = 0; i < exp.Length; i++) if (mask[i] != 0) exp[i] = pmVal[i % pmVal.Length];
              select.Add(Case($"putmask/decimal/c/{n++}", "putmask", "{}",
                new[] { SvC().Describe(), BoolOperandDesc(mask, svShape),
                        new Operand { Base = pmVal, Shape = new[] { pmVal.Length }, Strides = new long[] { 1 }, Offset = 0 }.Describe() },
                "decimal", svShape, HexOf(exp))); }
            // select (nc=1): cond ? choice : default. operands [cond, choice, default(0-d)].
            { var mask = sv.Select(x => (byte)(x > 0m ? 1 : 0)).ToArray(); decimal dflt = -99m;
              var exp = new decimal[sv.Length];
              for (int i = 0; i < exp.Length; i++) exp[i] = mask[i] != 0 ? sv2[i] : dflt;
              select.Add(Case($"select/decimal/c/{n++}", "select", "{\"nc\":1}",
                new[] { BoolOperandDesc(mask, svShape),
                        new Operand { Base = (decimal[])sv2.Clone(), Shape = svShape, Strides = new long[] { 1 }, Offset = 0 }.Describe(),
                        new Operand { Base = new[] { dflt }, Shape = new int[0], Strides = new long[0], Offset = 0 }.Describe() },
                "decimal", svShape, HexOf(exp))); }
            // choose (nc=2): out[i] = choices[index[i]][i]. index in {0,1}.
            { int[] cidx = { 0, 1, 0, 1, 1, 0, 1, 0 };
              var exp = new decimal[sv.Length];
              for (int i = 0; i < exp.Length; i++) exp[i] = cidx[i] == 0 ? sv[i] : sv2[i];
              select.Add(Case($"choose/decimal/c/{n++}", "choose", "{\"nc\":2}",
                new[] { Int32OperandDesc(cidx, svShape),
                        SvC().Describe(),
                        new Operand { Base = (decimal[])sv2.Clone(), Shape = svShape, Strides = new long[] { 1 }, Offset = 0 }.Describe() },
                "decimal", svShape, HexOf(exp))); }
            // compress (axis=0): a[cond]. operands [cond(bool), data]. cond = sv > 0.
            { var mask = sv.Select(x => (byte)(x > 0m ? 1 : 0)).ToArray();
              var exp = new List<decimal>(); for (int i = 0; i < sv.Length; i++) if (mask[i] != 0) exp.Add(sv[i]);
              select.Add(Case($"compress/decimal/c/{n++}", "compress", "{\"axis\":0}",
                new[] { BoolOperandDesc(mask, svShape), SvC().Describe() },
                "decimal", new[] { exp.Count }, HexOf(exp.ToArray()))); }
            // extract: a.flat[cond.flat]. operands [cond(bool), arr]. cond = sv != 0 (all nonzero here => full).
            { var mask = sv.Select(x => (byte)(x < 0m ? 1 : 0)).ToArray();   // negatives only
              var exp = new List<decimal>(); for (int i = 0; i < sv.Length; i++) if (mask[i] != 0) exp.Add(sv[i]);
              select.Add(Case($"extract/decimal/c/{n++}", "extract", "{}",
                new[] { BoolOperandDesc(mask, svShape), SvC().Describe() },
                "decimal", new[] { exp.Count }, HexOf(exp.ToArray()))); }
        }

        // ----- PRODUCTS: dot/inner/outer/vdot/tensordot/trace/kron — matmul was the ONLY covered
        // product, yet each of these routes through a DIFFERENT decimal accumulate/iterate path. -----
        {
            decimal[] a4 = { 1m, 2m, 3m, 4m }, b4 = { 5m, 6m, 7m, 8m };  // dot/inner/vdot/tensordot(1d)
            Operand V(decimal[] v) => new Operand { Base = (decimal[])v.Clone(), Shape = new[] { v.Length }, Strides = new long[] { 1 }, Offset = 0 };
            decimal dot4 = 0m; for (int i = 0; i < 4; i++) dot4 += a4[i] * b4[i];   // 70
            products.Add(Case($"dot/decimal/1d/{n++}", "dot", "{}", new[] { V(a4).Describe(), V(b4).Describe() }, "decimal", new int[0], HexOf(new[] { dot4 })));
            products.Add(Case($"inner/decimal/1d/{n++}", "inner", "{}", new[] { V(a4).Describe(), V(b4).Describe() }, "decimal", new int[0], HexOf(new[] { dot4 })));
            products.Add(Case($"vdot/decimal/1d/{n++}", "vdot", "{}", new[] { V(a4).Describe(), V(b4).Describe() }, "decimal", new int[0], HexOf(new[] { dot4 })));
            products.Add(Case($"tensordot/decimal/1d/{n++}", "tensordot", "{\"axes\":1}", new[] { V(a4).Describe(), V(b4).Describe() }, "decimal", new int[0], HexOf(new[] { dot4 })));
            // outer (na,nb)
            { decimal[] oa = { 1m, 2m, 3m }, ob = { 4m, 5m };
              var exp = new decimal[oa.Length * ob.Length];
              for (int i = 0; i < oa.Length; i++) for (int j = 0; j < ob.Length; j++) exp[i * ob.Length + j] = oa[i] * ob[j];
              products.Add(Case($"outer/decimal/{n++}", "outer", "{}", new[] { V(oa).Describe(), V(ob).Describe() }, "decimal", new[] { oa.Length, ob.Length }, HexOf(exp))); }
            // kron (1-D -> na*nb flat)
            { decimal[] ka = { 1m, 2m }, kb = { 3m, 4m, 5m };
              var exp = new decimal[ka.Length * kb.Length];
              for (int i = 0; i < ka.Length; i++) for (int j = 0; j < kb.Length; j++) exp[i * kb.Length + j] = ka[i] * kb[j];
              products.Add(Case($"kron/decimal/1d/{n++}", "kron", "{}", new[] { V(ka).Describe(), V(kb).Describe() }, "decimal", new[] { ka.Length * kb.Length }, HexOf(exp))); }
            // dot / tensordot 2-D @ 2-D (reuse the exact matmul), and vdot 2-D (flattened).
            { decimal[] A = { 1m, 2m, 3m, 4m, 5m, 6m }; int m = 2, k = 3, p2 = 2; decimal[] B = { 7m, 8m, 9m, 10m, 11m, 12m };
              var exp = MatMul2D(A, m, k, B, p2);
              var Aop = new Operand { Base = A, Shape = new[] { m, k }, Strides = CStrides(new[] { m, k }), Offset = 0 };
              var Bop = new Operand { Base = B, Shape = new[] { k, p2 }, Strides = CStrides(new[] { k, p2 }), Offset = 0 };
              products.Add(Case($"dot/decimal/2d/{n++}", "dot", "{}", new[] { Aop.Describe(), Bop.Describe() }, "decimal", new[] { m, p2 }, HexOf(exp)));
              products.Add(Case($"tensordot/decimal/2d/{n++}", "tensordot", "{\"axes\":1}", new[] { Aop.Describe(), Bop.Describe() }, "decimal", new[] { m, p2 }, HexOf(exp))); }
            { decimal[] A = { 1m, 2m, 3m, 4m }, B = { 5m, 6m, 7m, 8m };   // vdot flattens both -> scalar
              decimal vd = 0m; for (int i = 0; i < 4; i++) vd += A[i] * B[i];
              var Aop = new Operand { Base = A, Shape = new[] { 2, 2 }, Strides = CStrides(new[] { 2, 2 }), Offset = 0 };
              var Bop = new Operand { Base = B, Shape = new[] { 2, 2 }, Strides = CStrides(new[] { 2, 2 }), Offset = 0 };
              products.Add(Case($"vdot/decimal/2d/{n++}", "vdot", "{}", new[] { Aop.Describe(), Bop.Describe() }, "decimal", new int[0], HexOf(new[] { vd }))); }
            // trace: square (3,3) and non-square (2,3) — sum of the main diagonal.
            { decimal[] T = { 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m };
              var Top = new Operand { Base = T, Shape = new[] { 3, 3 }, Strides = CStrides(new[] { 3, 3 }), Offset = 0 };
              products.Add(Case($"trace/decimal/3x3/{n++}", "trace", "{}", new[] { Top.Describe() }, "decimal", new int[0], HexOf(new[] { T[0] + T[4] + T[8] }))); }
            { decimal[] T = { 1m, 2m, 3m, 4m, 5m, 6m };
              var Top = new Operand { Base = T, Shape = new[] { 2, 3 }, Strides = CStrides(new[] { 2, 3 }), Offset = 0 };
              products.Add(Case($"trace/decimal/2x3/{n++}", "trace", "{}", new[] { Top.Describe() }, "decimal", new int[0], HexOf(new[] { T[0] + T[4] }))); }
        }

        // ----- SEARCH: argsort/searchsorted/unique/nonzero/flatnonzero/lexsort — the decimal
        // compare-driven sort/search paths (the same 16-byte-compare hazard the flat-argmax bug hit). -----
        {
            decimal[] dist = { 3m, -1m, 4m, 1m, -5m, 9m, 2m, 6m };            // distinct (unambiguous argsort)
            var dShape = new[] { dist.Length };
            Operand D() => new Operand { Base = (decimal[])dist.Clone(), Shape = dShape, Strides = new long[] { 1 }, Offset = 0 };
            search.Add(Case($"argsort/decimal/1d/{n++}", "argsort", "{\"axis\":0}", new[] { D().Describe() },
                "int64", dShape, HexOf(Bytes(ArgsortStable(dist)))));
            // negstride source (same logical values) — exercises argsort over a reversed 16-byte view.
            { var b = (decimal[])dist.Clone(); Array.Reverse(b);
              var neg = new Operand { Base = b, Shape = dShape, Strides = new long[] { -1 }, Offset = dist.Length - 1 };
              search.Add(Case($"argsort/decimal/negstride/{n++}", "argsort", "{\"axis\":0}", new[] { neg.Describe() },
                "int64", dShape, HexOf(Bytes(ArgsortStable(dist))))); }
            // searchsorted: sorted haystack + arbitrary needles, both sides.
            { decimal[] sorted = { -5m, -1m, 1m, 2m, 3m, 4m, 6m, 9m }; decimal[] vals = { 0m, 3m, 10m, -6m, 2m };
              var sOp = new Operand { Base = sorted, Shape = new[] { sorted.Length }, Strides = new long[] { 1 }, Offset = 0 };
              var vOp = new Operand { Base = vals, Shape = new[] { vals.Length }, Strides = new long[] { 1 }, Offset = 0 };
              foreach (var side in new[] { "left", "right" })
                search.Add(Case($"searchsorted/decimal/{side}/{n++}", "searchsorted", $"{{\"side\":\"{side}\"}}",
                    new[] { sOp.Describe(), vOp.Describe() }, "int64", new[] { vals.Length },
                    HexOf(Bytes(SearchSortedIdx(sorted, vals, side == "right"))))); }
            // unique: sorted distinct over a duplicate-laced source.
            { decimal[] dup = { 3m, 1m, 3m, -2m, 1m, 5m, -2m, 3m };
              var u = UniqueSorted(dup);
              var uOp = new Operand { Base = dup, Shape = new[] { dup.Length }, Strides = new long[] { 1 }, Offset = 0 };
              search.Add(Case($"unique/decimal/1d/{n++}", "unique", "{}", new[] { uOp.Describe() }, "decimal", new[] { u.Length }, HexOf(u))); }
            // nonzero()[0] and flatnonzero over a zero-laced source.
            { decimal[] z = { 0m, 3m, 0m, -2m, 0m, 0m, 5m, 1m };
              var idx = NonzeroIdx(z);
              var zOp = new Operand { Base = z, Shape = new[] { z.Length }, Strides = new long[] { 1 }, Offset = 0 };
              search.Add(Case($"nonzero/decimal/1d/{n++}", "nonzero", "{}", new[] { zOp.Describe() }, "int64", new[] { idx.Length }, HexOf(Bytes(idx))));
              search.Add(Case($"flatnonzero/decimal/1d/{n++}", "flatnonzero", "{}", new[] { zOp.Describe() }, "int64", new[] { idx.Length }, HexOf(Bytes(idx)))); }
            // lexsort: single key (== argsort) and two keys (primary last; ties broken by secondary).
            search.Add(Case($"lexsort/decimal/1key/{n++}", "lexsort", "{\"axis\":-1}", new[] { D().Describe() },
                "int64", dShape, HexOf(Bytes(ArgsortStable(dist)))));
            { decimal[] primary = { 1m, 1m, 2m, 2m, 0m, 0m }, secondary = { 5m, 1m, 9m, 2m, 7m, 3m };
              // np.lexsort(keys) sorts by the LAST key first: OpRegistry passes ops=[secondary, primary].
              var order = Enumerable.Range(0, primary.Length)
                                    .OrderBy(i => primary[i]).ThenBy(i => secondary[i])
                                    .Select(i => (long)i).ToArray();
              var pOp = new Operand { Base = primary, Shape = new[] { primary.Length }, Strides = new long[] { 1 }, Offset = 0 };
              var sOp = new Operand { Base = secondary, Shape = new[] { secondary.Length }, Strides = new long[] { 1 }, Offset = 0 };
              search.Add(Case($"lexsort/decimal/2key/{n++}", "lexsort", "{\"axis\":-1}",
                new[] { sOp.Describe(), pOp.Describe() }, "int64", new[] { primary.Length }, HexOf(Bytes(order)))); }
        }

        // ----- MANIP EXTEND: flip/roll/concatenate/stack/hstack/vstack/squeeze/moveaxis/swapaxes/
        // expand_dims/broadcast_to/pad/append/insert/delete/repeat/tile/ediff1d. Value-preserving
        // reindex/copy — ravel/transpose/reshape already cover the core materialize path; these carry
        // their own slab-copy / stride logic. Appended to the manip corpus. -----
        {
            decimal[] mv = { 1m, -2m, 3m, -4m, 5m, 6m };
            decimal[] nv = { 10m, 20m, 30m, 40m, 50m, 60m };
            Operand M1() => new Operand { Base = (decimal[])mv.Clone(), Shape = new[] { 6 }, Strides = new long[] { 1 }, Offset = 0 };
            Operand N1() => new Operand { Base = (decimal[])nv.Clone(), Shape = new[] { 6 }, Strides = new long[] { 1 }, Offset = 0 };
            Operand M23() => new Operand { Base = (decimal[])mv.Clone(), Shape = new[] { 2, 3 }, Strides = CStrides(new[] { 2, 3 }), Offset = 0 };
            Operand N23() => new Operand { Base = (decimal[])nv.Clone(), Shape = new[] { 2, 3 }, Strides = CStrides(new[] { 2, 3 }), Offset = 0 };

            // flip 1-D (reverse), flip 2-D all-axes / axis0 / axis1.
            manip.Add(Case($"flip/decimal/1d/{n++}", "flip", "{}", new[] { M1().Describe() }, "decimal", new[] { 6 },
                HexOf(mv.Reverse().ToArray())));
            { var e = new decimal[6]; for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) e[i * 3 + j] = mv[(1 - i) * 3 + (2 - j)];
              manip.Add(Case($"flip/decimal/2d/all/{n++}", "flip", "{}", new[] { M23().Describe() }, "decimal", new[] { 2, 3 }, HexOf(e))); }
            { var e = new decimal[6]; for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) e[i * 3 + j] = mv[(1 - i) * 3 + j];
              manip.Add(Case($"flip/decimal/2d/ax0/{n++}", "flip", "{\"axis\":0}", new[] { M23().Describe() }, "decimal", new[] { 2, 3 }, HexOf(e))); }
            { var e = new decimal[6]; for (int i = 0; i < 2; i++) for (int j = 0; j < 3; j++) e[i * 3 + j] = mv[i * 3 + (2 - j)];
              manip.Add(Case($"flip/decimal/2d/ax1/{n++}", "flip", "{\"axis\":1}", new[] { M23().Describe() }, "decimal", new[] { 2, 3 }, HexOf(e))); }
            // roll (flat) by shift 2.
            { int sh = 2; var e = new decimal[6]; for (int i = 0; i < 6; i++) e[i] = mv[((i - sh) % 6 + 6) % 6];
              manip.Add(Case($"roll/decimal/1d/{n++}", "roll", "{\"shift\":2}", new[] { M1().Describe() }, "decimal", new[] { 6 }, HexOf(e))); }
            // concatenate 1-D (axis 0), 2-D axis 0 / axis 1.
            manip.Add(Case($"concatenate/decimal/1d/{n++}", "concatenate", "{\"axis\":0}", new[] { M1().Describe(), N1().Describe() },
                "decimal", new[] { 12 }, HexOf(mv.Concat(nv).ToArray())));
            manip.Add(Case($"concatenate/decimal/2d/ax0/{n++}", "concatenate", "{\"axis\":0}", new[] { M23().Describe(), N23().Describe() },
                "decimal", new[] { 4, 3 }, HexOf(mv.Concat(nv).ToArray())));
            { var e = new decimal[12]; // (2,3)|(2,3) along axis1 -> (2,6): row i = mv row i then nv row i
              for (int i = 0; i < 2; i++) { for (int j = 0; j < 3; j++) e[i * 6 + j] = mv[i * 3 + j]; for (int j = 0; j < 3; j++) e[i * 6 + 3 + j] = nv[i * 3 + j]; }
              manip.Add(Case($"concatenate/decimal/2d/ax1/{n++}", "concatenate", "{\"axis\":1}", new[] { M23().Describe(), N23().Describe() },
                "decimal", new[] { 2, 6 }, HexOf(e))); }
            // stack 1-D axis0 -> (2,6) ; axis1 -> (6,2).
            manip.Add(Case($"stack/decimal/ax0/{n++}", "stack", "{\"axis\":0}", new[] { M1().Describe(), N1().Describe() },
                "decimal", new[] { 2, 6 }, HexOf(mv.Concat(nv).ToArray())));
            { var e = new decimal[12]; for (int i = 0; i < 6; i++) { e[i * 2] = mv[i]; e[i * 2 + 1] = nv[i]; }
              manip.Add(Case($"stack/decimal/ax1/{n++}", "stack", "{\"axis\":1}", new[] { M1().Describe(), N1().Describe() },
                "decimal", new[] { 6, 2 }, HexOf(e))); }
            // hstack (1-D -> concat) ; vstack (1-D -> (2,6)).
            manip.Add(Case($"hstack/decimal/1d/{n++}", "hstack", "{}", new[] { M1().Describe(), N1().Describe() },
                "decimal", new[] { 12 }, HexOf(mv.Concat(nv).ToArray())));
            manip.Add(Case($"vstack/decimal/1d/{n++}", "vstack", "{}", new[] { M1().Describe(), N1().Describe() },
                "decimal", new[] { 2, 6 }, HexOf(mv.Concat(nv).ToArray())));
            // squeeze (1,6) -> (6,).
            { var s16 = new Operand { Base = (decimal[])mv.Clone(), Shape = new[] { 1, 6 }, Strides = CStrides(new[] { 1, 6 }), Offset = 0 };
              manip.Add(Case($"squeeze/decimal/{n++}", "squeeze", "{}", new[] { s16.Describe() }, "decimal", new[] { 6 }, HexOf(mv))); }
            // moveaxis / swapaxes on (2,3) -> transpose (3,2).
            manip.Add(Case($"moveaxis/decimal/{n++}", "moveaxis", "{\"src\":0,\"dst\":1}", new[] { M23().Describe() },
                "decimal", new[] { 3, 2 }, HexOf(TransposeReverse(mv, new[] { 2, 3 }))));
            manip.Add(Case($"swapaxes/decimal/{n++}", "swapaxes", "{\"a1\":0,\"a2\":1}", new[] { M23().Describe() },
                "decimal", new[] { 3, 2 }, HexOf(TransposeReverse(mv, new[] { 2, 3 }))));
            // expand_dims axis 0 -> (1,6) ; values unchanged.
            manip.Add(Case($"expand_dims/decimal/{n++}", "expand_dims", "{\"axis\":0}", new[] { M1().Describe() },
                "decimal", new[] { 1, 6 }, HexOf(mv)));
            // broadcast_to (6,) -> (2,6): each row = mv.
            manip.Add(Case($"broadcast_to/decimal/{n++}", "broadcast_to", "{\"shape\":[2,6]}", new[] { M1().Describe() },
                "decimal", new[] { 2, 6 }, HexOf(mv.Concat(mv).ToArray())));
            // pad (constant 0) width 1 -> [0, mv..., 0].
            { var e = new List<decimal> { 0m }; e.AddRange(mv); e.Add(0m);
              manip.Add(Case($"pad/decimal/{n++}", "pad", "{\"pad_width\":1,\"mode\":\"constant\"}", new[] { M1().Describe() },
                "decimal", new[] { 8 }, HexOf(e.ToArray()))); }
            // append (no axis -> flatten-append).
            manip.Add(Case($"append/decimal/{n++}", "append", "{}", new[] { M1().Describe(), N1().Describe() },
                "decimal", new[] { 12 }, HexOf(mv.Concat(nv).ToArray())));
            // insert one value before index 2 (axis 0).
            { decimal[] iv = { 99m }; var e = new List<decimal>(mv); e.Insert(2, iv[0]);
              manip.Add(Case($"insert/decimal/{n++}", "insert", "{\"obj\":2,\"axis\":0}",
                new[] { M1().Describe(), new Operand { Base = iv, Shape = new[] { 1 }, Strides = new long[] { 1 }, Offset = 0 }.Describe() },
                "decimal", new[] { 7 }, HexOf(e.ToArray()))); }
            // delete index 2 (axis 0).
            { var e = new List<decimal>(mv); e.RemoveAt(2);
              manip.Add(Case($"delete/decimal/{n++}", "delete", "{\"obj\":2,\"axis\":0}", new[] { M1().Describe() },
                "decimal", new[] { 5 }, HexOf(e.ToArray()))); }
            // repeat each element twice ; tile the array twice.
            { var e = new decimal[12]; for (int i = 0; i < 6; i++) { e[2 * i] = mv[i]; e[2 * i + 1] = mv[i]; }
              manip.Add(Case($"repeat/decimal/{n++}", "repeat", "{\"repeats\":2}", new[] { M1().Describe() }, "decimal", new[] { 12 }, HexOf(e))); }
            manip.Add(Case($"tile/decimal/{n++}", "tile", "{\"reps\":2}", new[] { M1().Describe() }, "decimal", new[] { 12 }, HexOf(mv.Concat(mv).ToArray())));
            // ediff1d: consecutive difference (len-1).
            { var e = new decimal[5]; for (int i = 0; i < 5; i++) e[i] = mv[i + 1] - mv[i];
              manip.Add(Case($"ediff1d/decimal/{n++}", "ediff1d", "{}", new[] { M1().Describe() }, "decimal", new[] { 5 }, HexOf(e))); }
        }

        // ----- AXIS GRANULARITY within already-covered families: the reduce tier had axis sum/min/max/
        // mean but NOT prod/argmax/argmin; scan/varstd/stat were flat-only. Add axis prod + argmax/argmin
        // (reduce), axis var/std (varstd), axis median/percentile/quantile (stat), axis cumsum/cumprod
        // (scan) over the three 2-D layouts × axis {0,1}. decimal + is exact, so accumulation order is
        // irrelevant; std=sqrt(var) is oracled by the same INDEPENDENT Newton DecSqrt as the flat tier. -----
        foreach (var ln in new[] { "c_contiguous_2d", "f_contiguous_2d", "strided_2d_cols" })
        {
            var o = SingleLayout(ln, 6, false);
            var log = o.Logical();
            int rows = o.Shape[0], cols = o.Shape[1];
            foreach (int axis in new[] { 0, 1 })
            {
                var slices = Slices2D(log, rows, cols, axis);
                int outN = slices.Length;
                // prod
                { var v = slices.Select(ProdSafe).ToArray();
                  foreach (var kd in new[] { false, true })
                    reduce.Add(Case($"prod/decimal/{ln}/ax{axis}kd{(kd ? 1 : 0)}/{n++}", "prod",
                        $"{{\"axis\":{axis},\"keepdims\":{(kd ? "true" : "false")}}}", new[] { o.Describe() },
                        "decimal", AxisOutShape(rows, cols, axis, outN, kd), HexOf(v))); }
                // argmax / argmin -> int64 index within each slice (first-occurrence on ties; pools distinct).
                { var amax = new long[outN]; var amin = new long[outN];
                  for (int s = 0; s < outN; s++)
                  { int im = 0, in_ = 0; var sl = slices[s];
                    for (int t = 1; t < sl.Length; t++) { if (sl[t] > sl[im]) im = t; if (sl[t] < sl[in_]) in_ = t; }
                    amax[s] = im; amin[s] = in_; }
                  reduce.Add(Case($"argmax/decimal/{ln}/ax{axis}/{n++}", "argmax", $"{{\"axis\":{axis},\"keepdims\":false}}",
                    new[] { o.Describe() }, "int64", new[] { outN }, HexOf(Bytes(amax))));
                  reduce.Add(Case($"argmin/decimal/{ln}/ax{axis}/{n++}", "argmin", $"{{\"axis\":{axis},\"keepdims\":false}}",
                    new[] { o.Describe() }, "int64", new[] { outN }, HexOf(Bytes(amin)))); }
                // var / std (ddof=0). The AXIS decimal var/std kernel (AxisVarStdDecimalHelper) computes
                // the mean and sum-of-squared-deviations in EXACT decimal but then rounds the variance
                // through a DOUBLE intermediate (`variance = (double)(sqDiffSum/divisor)`; std =
                // Math.Sqrt(variance)) before storing — UNLIKE the flat path (VarMomentsDecimal +
                // DecimalMath.Sqrt), which stays full-precision. So the axis result carries ~15
                // significant digits, not 28. The oracle replicates that EXACT double intermediate
                // (portable: Math.Sqrt is IEEE-correctly-rounded, the casts are deterministic) so it
                // still catches an iteration / mean / divisor bug (which diverges by orders of
                // magnitude) without over-claiming a precision the axis path does not provide — the
                // same per-path-fidelity policy the decimal TRANSCEND tier uses for its double bridge.
                { var vv = new decimal[outN]; var sd = new decimal[outN];
                  for (int s = 0; s < outN; s++)
                  { var sl = slices[s]; decimal mean = sl.Aggregate(0m, (x, y) => x + y) / sl.Length;
                    decimal sqDiff = sl.Aggregate(0m, (acc, x) => acc + (x - mean) * (x - mean));
                    double varD = (double)(sqDiff / (decimal)sl.Length);   // the kernel's double intermediate
                    vv[s] = (decimal)varD; sd[s] = (decimal)Math.Sqrt(varD); }
                  varstd.Add(Case($"var/decimal/{ln}/ax{axis}/{n++}", "var", $"{{\"axis\":{axis},\"keepdims\":false}}",
                    new[] { o.Describe() }, "decimal", new[] { outN }, HexOf(vv)));
                  varstd.Add(Case($"std/decimal/{ln}/ax{axis}/{n++}", "std", $"{{\"axis\":{axis},\"keepdims\":false}}",
                    new[] { o.Describe() }, "decimal", new[] { outN }, HexOf(sd))); }
                // median / percentile / quantile (per-slice order statistics)
                { var med = new decimal[outN];
                  for (int s = 0; s < outN; s++) { var sl = (decimal[])slices[s].Clone(); Array.Sort(sl); med[s] = Median(sl); }
                  stat.Add(Case($"median/decimal/{ln}/ax{axis}/{n++}", "median", $"{{\"axis\":{axis},\"keepdims\":false}}",
                    new[] { o.Describe() }, "decimal", new[] { outN }, HexOf(med)));
                  foreach (var (pq, frac) in new[] { ("25.0", 0.25), ("50.0", 0.5), ("75.0", 0.75) })
                  { var v = new decimal[outN];
                    for (int s = 0; s < outN; s++) { var sl = (decimal[])slices[s].Clone(); Array.Sort(sl); v[s] = Quantile(sl, frac); }
                    stat.Add(Case($"percentile/decimal/{ln}/ax{axis}/p{pq}/{n++}", "percentile", $"{{\"q\":{pq},\"axis\":{axis},\"keepdims\":false}}",
                        new[] { o.Describe() }, "decimal", new[] { outN }, HexOf(v))); }
                  foreach (var (qq, frac) in new[] { ("0.25", 0.25), ("0.5", 0.5), ("0.75", 0.75) })
                  { var v = new decimal[outN];
                    for (int s = 0; s < outN; s++) { var sl = (decimal[])slices[s].Clone(); Array.Sort(sl); v[s] = Quantile(sl, frac); }
                    stat.Add(Case($"quantile/decimal/{ln}/ax{axis}/q{qq}/{n++}", "quantile", $"{{\"q\":{qq},\"axis\":{axis},\"keepdims\":false}}",
                        new[] { o.Describe() }, "decimal", new[] { outN }, HexOf(v))); }
                }
                // cumsum / cumprod ALONG the axis (full-shape output).
                { var cs = new decimal[rows * cols]; var cp = new decimal[rows * cols];
                  if (axis == 0)
                    for (int j = 0; j < cols; j++)
                    { decimal s = 0m, p = 1m; for (int i = 0; i < rows; i++) { s += log[i * cols + j]; cs[i * cols + j] = s; p *= log[i * cols + j]; cp[i * cols + j] = p; } }
                  else
                    for (int i = 0; i < rows; i++)
                    { decimal s = 0m, p = 1m; for (int j = 0; j < cols; j++) { s += log[i * cols + j]; cs[i * cols + j] = s; p *= log[i * cols + j]; cp[i * cols + j] = p; } }
                  scan.Add(Case($"cumsum/decimal/{ln}/ax{axis}/{n++}", "cumsum", $"{{\"axis\":{axis}}}", new[] { o.Describe() }, "decimal", new[] { rows, cols }, HexOf(cs)));
                  scan.Add(Case($"cumprod/decimal/{ln}/ax{axis}/{n++}", "cumprod", $"{{\"axis\":{axis}}}", new[] { o.Describe() }, "decimal", new[] { rows, cols }, HexOf(cp))); }
            }
        }

        Write(Path.Combine(corpus, "decimal_unary.jsonl"), unary);
        Write(Path.Combine(corpus, "decimal_binary.jsonl"), binary);
        Write(Path.Combine(corpus, "decimal_reduce.jsonl"), reduce);
        Write(Path.Combine(corpus, "decimal_scan.jsonl"), scan);
        Write(Path.Combine(corpus, "decimal_power.jsonl"), power);
        Write(Path.Combine(corpus, "decimal_varstd.jsonl"), varstd);
        Write(Path.Combine(corpus, "decimal_matmul.jsonl"), matmul);
        Write(Path.Combine(corpus, "decimal_astype.jsonl"), astype);
        Write(Path.Combine(corpus, "decimal_stat.jsonl"), stat);
        Write(Path.Combine(corpus, "decimal_where.jsonl"), where);
        Write(Path.Combine(corpus, "decimal_sort.jsonl"), sort);
        Write(Path.Combine(corpus, "decimal_manip.jsonl"), manip);
        // G14 coverage-audit expansion (new files; the extended binary/reduce/scan/varstd/stat/manip
        // rows land in their existing files above).
        Write(Path.Combine(corpus, "decimal_extra.jsonl"), extra);
        Write(Path.Combine(corpus, "decimal_transcend.jsonl"), transcend);
        Write(Path.Combine(corpus, "decimal_select.jsonl"), select);
        Write(Path.Combine(corpus, "decimal_products.jsonl"), products);
        Write(Path.Combine(corpus, "decimal_search.jsonl"), search);
    }

    // small in-range decimal source for decimal->narrow casts (truncation toward zero).
    static Operand SmallSingle(string ln)
    {
        decimal[] sp = { 0m, 1m, -1m, 2.75m, -2.75m, 7.9m, -7.9m, 42.5m, -3.2m, 100m, -100m, 5m, 9.99m, -9.99m, 3m, 8m };
        decimal[] B(int n) { var a = new decimal[n]; for (int i=0;i<n;i++) a[i]=sp[i%sp.Length]; return a; }
        switch (ln)
        {
            case "c_contiguous_1d": return new Operand { Base = B(8), Shape = new[]{8}, Strides = new long[]{1}, Offset = 0 };
            case "c_contiguous_2d": return new Operand { Base = B(20), Shape = new[]{4,5}, Strides = new long[]{5,1}, Offset = 0 };
            case "strided_step2_1d": return new Operand { Base = B(16), Shape = new[]{8}, Strides = new long[]{2}, Offset = 0 };
            case "negstride_1d": return new Operand { Base = B(8), Shape = new[]{8}, Strides = new long[]{-1}, Offset = 7 };
            default: throw new Exception("small layout " + ln);
        }
    }

    static string AstypeTo(string dtype, int[] shape, byte[] srcBuf, string expDecHex, ref int n)
    {
        // operand is the SOURCE dtype; expected is decimal.
        long bufN = shape.Aggregate(1,(a,b)=>a*b);
        string op = $"{{\"dtype\":\"decimal\"}}";
        string operand = $"{{\"dtype\":\"{dtype}\",\"shape\":[{string.Join(",",shape)}],\"strides\":[{string.Join(",",CStrides(shape))}],\"offset\":0,\"bufferSize\":{bufN},\"buffer\":\"{HexOf(srcBuf)}\"}}";
        return $"{{\"id\":\"astype/{dtype}->decimal/{n++}\",\"op\":\"astype\",\"params\":{op},\"operands\":[{operand}],\"expected\":{{\"dtype\":\"decimal\",\"shape\":[{string.Join(",",shape)}],\"buffer\":\"{expDecHex}\"}},\"layout\":\"decimal\",\"valueclass\":\"decimal\"}}";
    }

    static decimal IntPow(decimal a, int e)
    {
        if (e == 0) return 1m;
        bool neg = e < 0; int n = Math.Abs(e);
        decimal r = 1m; for (int i = 0; i < n; i++) r *= a;
        return neg ? 1m / r : r;
    }

    // Independent decimal sqrt (Newton-Raphson) — the oracle for std (NOT NumSharp's DecimalMath.Sqrt).
    static decimal DecSqrt(decimal x)
    {
        if (x <= 0m) return 0m;
        decimal g = (decimal)Math.Sqrt((double)x);   // seed from double
        if (g <= 0m) g = 1m;
        for (int i = 0; i < 40; i++) { decimal ng = (g + x / g) / 2m; if (ng == g) break; g = ng; }
        return g;
    }

    // median of an ALREADY-SORTED decimal[] (even n -> exact average of the two middles).
    static decimal Median(decimal[] sorted)
    {
        int n = sorted.Length;
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2m;
    }

    // NumPy 'linear' quantile of an ALREADY-SORTED decimal[]: index = frac*(n-1), interpolate.
    static decimal Quantile(decimal[] sorted, double frac)
    {
        int n = sorted.Length;
        if (n == 1) return sorted[0];
        double idx = frac * (n - 1);
        int lo = (int)Math.Floor(idx), hi = (int)Math.Ceiling(idx);
        if (lo == hi) return sorted[lo];
        decimal w = (decimal)(idx - lo);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * w;
    }

    // sort each 1-D slice along `axis` (negative axis normalized) — independent Array.Sort oracle.
    static decimal[] SortAxis(decimal[] flat, int[] shape, int axis)
    {
        if (axis < 0) axis += shape.Length;
        var outv = (decimal[])flat.Clone();
        var strides = CStrides(shape);
        long axLen = shape[axis], axStride = strides[axis];
        int[] outer = (int[])shape.Clone(); outer[axis] = 1;
        int outerN = outer.Aggregate(1, (a, b) => a * b);
        var coord = new int[shape.Length];
        for (int o = 0; o < outerN; o++)
        {
            long baseF = 0; for (int d = 0; d < shape.Length; d++) baseF += coord[d] * strides[d];
            var slice = new decimal[axLen];
            for (long k = 0; k < axLen; k++) slice[k] = outv[baseF + k * axStride];
            Array.Sort(slice);
            for (long k = 0; k < axLen; k++) outv[baseF + k * axStride] = slice[k];
            for (int d = shape.Length - 1; d >= 0; d--) { if (d == axis) continue; if (++coord[d] < shape[d]) break; coord[d] = 0; }
        }
        return outv;
    }

    // repeated consecutive difference along `axis`, applied nDiff times (a[1:]-a[:-1]).
    static decimal[] DiffAxis(decimal[] flat, int[] shape, int nDiff, int axis)
    {
        if (axis < 0) axis += shape.Length;
        var cur = flat; var curShape = (int[])shape.Clone();
        for (int it = 0; it < nDiff; it++)
        {
            var strides = CStrides(curShape);
            var newShape = (int[])curShape.Clone(); newShape[axis] -= 1;
            int newN = newShape.Aggregate(1, (a, b) => a * b);
            var res = new decimal[newN];
            var coord = new int[newShape.Length];
            for (int i = 0; i < newN; i++)
            {
                long srcLo = 0, srcHi = 0;
                for (int d = 0; d < curShape.Length; d++) { int c = coord[d]; srcLo += c * strides[d]; srcHi += (d == axis ? c + 1 : c) * strides[d]; }
                res[i] = cur[srcHi] - cur[srcLo];
                Inc(coord, newShape);
            }
            cur = res; curShape = newShape;
        }
        return cur;
    }

    // np.transpose with no axes = reverse all axes; return the result's C-order values.
    static decimal[] TransposeReverse(decimal[] flat, int[] shape)
    {
        int nd = shape.Length;
        var newShape = RevShape(shape);
        var strides = CStrides(shape);
        var outv = new decimal[flat.Length];
        var coord = new int[nd];
        for (int i = 0; i < flat.Length; i++)
        {
            long srcF = 0;
            for (int d = 0; d < nd; d++) srcF += coord[d] * strides[nd - 1 - d];
            outv[i] = flat[srcF];
            Inc(coord, newShape);
        }
        return outv;
    }

    static int[] RevShape(int[] shape) { var r = new int[shape.Length]; for (int i = 0; i < shape.Length; i++) r[i] = shape[shape.Length - 1 - i]; return r; }

    // largest factor of n that is <= sqrt(n) (for a non-trivial 2-D reshape target).
    static int LargestFactorLeqSqrt(int n)
    {
        int best = 1;
        for (int f = 2; f * f <= n; f++) if (n % f == 0) best = f;
        return best;
    }

    // bool operand descriptor (C-contiguous mask) — the where() condition.
    static string BoolOperandDesc(byte[] mask, int[] shape)
    {
        var sb = new StringBuilder(mask.Length * 2); foreach (var b in mask) sb.Append(b.ToString("x2"));
        return $"{{\"dtype\":\"bool\",\"shape\":[{string.Join(",", shape)}],"
             + $"\"strides\":[{string.Join(",", CStrides(shape))}],\"offset\":0,"
             + $"\"bufferSize\":{mask.Length},\"buffer\":\"{sb}\"}}";
    }

    static byte[] Bytes(long[] v) { var b = new byte[v.Length*8]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
    static byte[] Bytes(short[] v) { var b = new byte[v.Length*2]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
    static byte[] Bytes(ulong[] v) { var b = new byte[v.Length*8]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
    static byte[] Bytes(int[] v) { var b = new byte[v.Length*4]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
    static byte[] Bytes(double[] v) { var b = new byte[v.Length*8]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
    static byte[] Bytes(float[] v) { var b = new byte[v.Length*4]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }

    static decimal ProdSafe(decimal[] v) { decimal p = 1m; foreach (var x in v) p *= x; return p; }

    // ============================ G14 coverage-audit helpers ============================
    // Each computes an expected value with NAIVE scalar System.Decimal (or the EXACT double bridge the
    // decimal kernel uses for transcendentals — DirectILKernelGenerator.Unary.Decimal.cs) — never a
    // NumSharp kernel. BitDiff tokenizes decimal by canonical VALUE (trailing zeros stripped, -0 == 0),
    // so scale variants and sign-of-zero never false-fail; only wrong VALUES / wrong iteration do.

    /// <summary>C-contiguous int64 operand descriptor — index arrays for take/take_along_axis/lexsort
    /// (int64 is same_kind/safe-castable so every selection op accepts it).</summary>
    static string Int64OperandDesc(long[] v, int[] shape)
        => $"{{\"dtype\":\"int64\",\"shape\":[{string.Join(",", shape)}],"
         + $"\"strides\":[{string.Join(",", CStrides(shape))}],\"offset\":0,"
         + $"\"bufferSize\":{v.Length},\"buffer\":\"{HexOf(Bytes(v))}\"}}";

    /// <summary>C-contiguous int32 operand descriptor — the put/choose index arrays (np.array(int[])
    /// is int32, the common call form; int32 casts same_kind to intp for take and safe for put).</summary>
    static string Int32OperandDesc(int[] v, int[] shape)
        => $"{{\"dtype\":\"int32\",\"shape\":[{string.Join(",", shape)}],"
         + $"\"strides\":[{string.Join(",", CStrides(shape))}],\"offset\":0,"
         + $"\"bufferSize\":{v.Length},\"buffer\":\"{HexOf(Bytes(v))}\"}}";

    /// <summary>10^k as an exact decimal (k>=0) — round_'s scale factor for negative decimals.</summary>
    static decimal Pow10(int k) { decimal p = 1m; for (int i = 0; i < k; i++) p *= 10m; return p; }

    /// <summary>np.round_ / np.around for decimal: banker's (half-to-even) at `decimals` places.
    /// decimals&gt;=0 is Math.Round(x, decimals); decimals&lt;0 rounds to tens/hundreds via scale-round-
    /// unscale (matches the probed kernel: around(125,-1)=120, around(-2.5,-1)=-0m≡0m).</summary>
    static decimal RoundDec(decimal x, int decimals)
    {
        if (decimals >= 0) return Math.Round(x, decimals, MidpointRounding.ToEven);
        decimal s = Pow10(-decimals);
        return Math.Round(x / s, 0, MidpointRounding.ToEven) * s;
    }

    /// <summary>np.spacing(decimal) — the kernel's (decimal)npy_spacing((double)x) bridge. Pure bit
    /// arithmetic on the double (no libm), so this is PORTABLE and rides the strict tier. A decimal is
    /// always finite, so the inf/NaN branches of npy_spacing are unreachable.</summary>
    static decimal SpacingDec(decimal x)
    {
        double d = (double)x;
        double sp = d == 0.0 ? double.Epsilon
                  : BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(d) + 1L) - d;
        return (decimal)sp;   // d finite => sp finite and within decimal range
    }

    /// <summary>fmod (C fmod / truncated remainder): sign follows the DIVIDEND, unlike the floored
    /// np.mod. C# decimal `%` IS truncated remainder, so this is a % b (probed: fmod(-7,3)=-1).</summary>
    static decimal Fmod(decimal a, decimal b) => a % b;

    /// <summary>np.heaviside(x1, x2): 0 for x1&lt;0, x2 for x1==0, 1 for x1&gt;0.</summary>
    static decimal Heaviside(decimal x1, decimal x2) => x1 < 0m ? 0m : (x1 == 0m ? x2 : 1m);

    /// <summary>np.copysign(a, b): |a| with the sign of b. b's sign is Math.Sign (a +0/-0 decimal both
    /// read non-negative); the -0m produced by copysign(0,-x) ties 0m under BitDiff.</summary>
    static decimal Copysign(decimal a, decimal b) => Math.Sign(b) < 0 ? -Math.Abs(a) : Math.Abs(a);

    /// <summary>Stable ascending argsort (ties broken by original position), int64 positions — matches
    /// NumSharp's stable decimal argsort. Pools used are distinct-valued so ties do not arise.</summary>
    static long[] ArgsortStable(decimal[] a)
    {
        var idx = new long[a.Length];
        for (int i = 0; i < a.Length; i++) idx[i] = i;
        Array.Sort(idx, (x, y) => { int c = a[x].CompareTo(a[y]); return c != 0 ? c : x.CompareTo(y); });
        return idx;
    }

    /// <summary>np.searchsorted insertion points (int64): leftmost (right=false) / rightmost
    /// (right=true) index keeping `sorted` sorted after inserting each value.</summary>
    static long[] SearchSortedIdx(decimal[] sorted, decimal[] vals, bool right)
    {
        var r = new long[vals.Length];
        for (int k = 0; k < vals.Length; k++)
        {
            int lo = 0, hi = sorted.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                bool goRight = right ? sorted[mid] <= vals[k] : sorted[mid] < vals[k];
                if (goRight) lo = mid + 1; else hi = mid;
            }
            r[k] = lo;
        }
        return r;
    }

    /// <summary>Sorted distinct values — np.unique over a flat decimal view (the sort path decimal takes).</summary>
    static decimal[] UniqueSorted(decimal[] a)
    {
        var s = (decimal[])a.Clone(); Array.Sort(s);
        var outv = new List<decimal>();
        for (int i = 0; i < s.Length; i++) if (i == 0 || s[i] != s[i - 1]) outv.Add(s[i]);
        return outv.ToArray();
    }

    /// <summary>Flat int64 positions where the value is non-zero (np.flatnonzero / np.nonzero()[0]).</summary>
    static long[] NonzeroIdx(decimal[] a)
    {
        var outv = new List<long>();
        for (int i = 0; i < a.Length; i++) if (a[i] != 0m) outv.Add((long)i);
        return outv.ToArray();
    }

    /// <summary>Exact 2-D decimal matrix product (m,k)@(k,p) -> (m,p) C-order. decimal + is exact, so
    /// the accumulation ORDER is irrelevant — any kernel order gives byte-identical values.</summary>
    static decimal[] MatMul2D(decimal[] a, int m, int k, decimal[] b, int p)
    {
        var outv = new decimal[m * p];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < p; j++)
            { decimal acc = 0m; for (int t = 0; t < k; t++) acc += a[i * k + t] * b[t * p + j]; outv[i * p + j] = acc; }
        return outv;
    }

    /// <summary>The 1-D slices along `axis` of a 2-D (rows,cols) C-order array, in output order
    /// (one per column for axis 0, one per row for axis 1) — the building block for the axis reduce/
    /// scan/stat expected values below.</summary>
    static decimal[][] Slices2D(decimal[] log, int rows, int cols, int axis)
    {
        int outN = axis == 0 ? cols : rows, m = axis == 0 ? rows : cols;
        var res = new decimal[outN][];
        for (int s = 0; s < outN; s++)
        {
            var slice = new decimal[m];
            for (int t = 0; t < m; t++) slice[t] = axis == 0 ? log[t * cols + s] : log[s * cols + t];
            res[s] = slice;
        }
        return res;
    }

    /// <summary>The keepdims-aware output shape for a 2-D axis reduction to `outN` values: (outN,) when
    /// keepdims is false, else (1,cols) for axis 0 / (rows,1) for axis 1.</summary>
    static int[] AxisOutShape(int rows, int cols, int axis, int outN, bool keepdims)
        => !keepdims ? new[] { outN } : (axis == 0 ? new[] { 1, cols } : new[] { rows, 1 });

    static decimal[] BroadcastApply(decimal[] la, int[] sa, decimal[] lb, int[] sb, int[] rs, Func<decimal,decimal,decimal> f)
    {
        int n = rs.Aggregate(1,(x,y)=>x*y); var outv = new decimal[n];
        var coord = new int[rs.Length];
        for (int i = 0; i < n; i++) { outv[i] = f(At(la, sa, coord), At(lb, sb, coord)); Inc(coord, rs); }
        return outv;
    }
    static byte[] BroadcastApplyBool(decimal[] la, int[] sa, decimal[] lb, int[] sb, int[] rs, Func<decimal,decimal,bool> f)
    {
        int n = rs.Aggregate(1,(x,y)=>x*y); var outv = new byte[n];
        var coord = new int[rs.Length];
        for (int i = 0; i < n; i++) { outv[i] = (byte)(f(At(la, sa, coord), At(lb, sb, coord)) ? 1 : 0); Inc(coord, rs); }
        return outv;
    }
    static void Inc(int[] coord, int[] shp) { for (int d = shp.Length - 1; d >= 0; d--) { if (++coord[d] < shp[d]) break; coord[d] = 0; } }

    static void Write(string path, List<string> lines)
    {
        File.WriteAllText(path, string.Join("\n", lines) + (lines.Count > 0 ? "\n" : ""), new UTF8Encoding(false));
        Console.WriteLine($"wrote {lines.Count} cases -> {path}");
    }

    static string FindOracleDir()
    {
        // this file lives in test/oracle/; AppContext.BaseDirectory is the build temp, so search upward
        // from the current directory for a folder containing NumSharp.Tests.
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir != null; i++)
        {
            var cand = Path.Combine(dir, "test", "oracle");
            if (Directory.Exists(Path.Combine(dir, "test", "NumSharp.Tests"))) return cand;
            if (Path.GetFileName(dir) == "oracle" && Directory.Exists(Path.Combine(dir, "..", "NumSharp.Tests"))) return dir;
            dir = Path.GetDirectoryName(dir);
        }
        // fallback: assume invoked from test/oracle
        return Directory.GetCurrentDirectory();
    }
}
