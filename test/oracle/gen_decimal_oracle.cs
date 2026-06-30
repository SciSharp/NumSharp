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
            default: throw new Exception("unknown pair layout " + name);
        }
    }
    static readonly string[] PAIR_LAYOUTS = {
        "pp_contig_contig","pp_contig_fortran","pp_strided_strided","pp_scalar_right",
        "pp_scalar_left","pp_broadcast_row","pp_negstride_both",
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
        }

        var power = new List<string>();
        var varstd = new List<string>();
        var matmul = new List<string>();
        var astype = new List<string>();

        // ----- POWER decimal^int (exact: repeated multiply / reciprocal). Exponent is a 0-D
        // decimal whose value is a whole number — DecimalMath.Pow must be exact for integer powers. -----
        foreach (var ln in new[] { "c_contiguous_1d", "c_contiguous_2d", "strided_step2_1d", "negstride_1d", "broadcast_1d_to_2d" })
        {
            foreach (int e in new[] { 0, 1, 2, 3 })
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

        Write(Path.Combine(corpus, "decimal_unary.jsonl"), unary);
        Write(Path.Combine(corpus, "decimal_binary.jsonl"), binary);
        Write(Path.Combine(corpus, "decimal_reduce.jsonl"), reduce);
        Write(Path.Combine(corpus, "decimal_scan.jsonl"), scan);
        Write(Path.Combine(corpus, "decimal_power.jsonl"), power);
        Write(Path.Combine(corpus, "decimal_varstd.jsonl"), varstd);
        Write(Path.Combine(corpus, "decimal_matmul.jsonl"), matmul);
        Write(Path.Combine(corpus, "decimal_astype.jsonl"), astype);
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

    static byte[] Bytes(long[] v) { var b = new byte[v.Length*8]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
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
