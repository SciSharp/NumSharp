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
        // resolve test/oracle dir regardless of run cwd: walk up to find NumSharp.UnitTest sibling.
        string oracleDir = FindOracleDir();
        string corpus = Path.GetFullPath(Path.Combine(oracleDir, "..", "NumSharp.UnitTest", "Fuzz", "corpus"));
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
        // from the current directory for a folder containing NumSharp.UnitTest.
        var dir = Directory.GetCurrentDirectory();
        for (int i = 0; i < 8 && dir != null; i++)
        {
            var cand = Path.Combine(dir, "test", "oracle");
            if (Directory.Exists(Path.Combine(dir, "test", "NumSharp.UnitTest"))) return cand;
            if (Path.GetFileName(dir) == "oracle" && Directory.Exists(Path.Combine(dir, "..", "NumSharp.UnitTest"))) return dir;
            dir = Path.GetDirectoryName(dir);
        }
        // fallback: assume invoked from test/oracle
        return Directory.GetCurrentDirectory();
    }
}
