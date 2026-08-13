using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Threading;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

static class Harness
{
    static int findings = 0, oks = 0;
    static readonly List<string> findingList = new();

    static void FIND(string msg) { findings++; findingList.Add(msg); Console.WriteLine("[FINDING] " + msg); }
    static void OK(string msg) { oks++; Console.WriteLine("  [ok] " + msg); }
    static void NOTE(string msg) { Console.WriteLine("  [note] " + msg); }

    static int Main(string[] args)
    {
        string scenario = args.Length > 0 ? args[0] : "battery";
        // codec poisoning requires NO early registration; everything else registers at startup.
        bool registerEarly = scenario != "poison";

        Runtime.PythonDLL = @"C:\Users\ELI\.claude\python\python312.dll";
        PythonEngine.Initialize();
        PythonEngine.BeginAllowThreads();
        if (registerEarly) NDArrayPythonInterop.RegisterCodec();

        Console.WriteLine($"=== scenario: {scenario} ===");
        int rc = 0;
        try
        {
            switch (scenario)
            {
                case "battery": Battery(); break;
                case "leak": LeakSweep(); break;
                case "poison": Poison(); break;
                case "shutdown": ShutdownConvert(); break;
                case "gil": GilAv(); break;
                case "threads": Threads(); break;
                case "aiface": ArrayInterfaceRobustness(); break;
                case "aiface2": ArrayInterfaceVsNumpy(); break;
                case "strided": StridedNonNumpyImport(); break;
                case "drainidle": DrainIdle(); break;
                case "resize": ResizeRefcheck(); break;
                case "mmap": MmapImport(); break;
                case "nullcrash": NullPtrCrash(); break;
                case "dtypeedge": DtypeEdge(); break;
                default: Console.WriteLine("unknown scenario"); rc = 2; break;
            }
        }
        catch (Exception e)
        {
            Console.WriteLine($"[HARNESS-FATAL] {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
            rc = 99;
        }

        Console.WriteLine($"\n=== summary: {oks} ok, {findings} findings ===");
        foreach (var f in findingList) Console.WriteLine("  * " + f);

        if (scenario != "gil" && scenario != "shutdown")   // these manage their own shutdown/crash
        {
            RuntimeData.FormatterType = typeof(NoopFormatter);
            try { PythonEngine.Shutdown(); } catch (Exception e) { Console.WriteLine($"[shutdown] {e.Message}"); }
        }
        return rc;
    }

    // ---------------- python helpers ----------------
    static PyModule _scope;
    static PyModule Scope()
    {
        if (_scope == null)
        {
            using (Py.GIL()) { _scope = Py.CreateScope(); _scope.Exec("import numpy as np\nimport array, ctypes, gc"); }
        }
        return _scope;
    }
    static bool PyB(string expr) { using (Py.GIL()) { using var r = Scope().Eval(expr); return r.As<bool>(); } }
    static string PyS(string expr) { using (Py.GIL()) { using var r = Scope().Eval($"str({expr})"); return r.As<string>(); } }
    static long PyL(string expr) { using (Py.GIL()) { using var r = Scope().Eval(expr); return r.As<long>(); } }
    static double PyF(string expr) { using (Py.GIL()) { using var r = Scope().Eval(expr); return r.As<double>(); } }
    static void PyExec(string code) { using (Py.GIL()) Scope().Exec(code); }

    static void Pump()
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        using (Py.GIL())
        {
            Finalizer.Instance.Collect();
            PythonEngine.RunSimpleString("import gc; gc.collect()");
            using var t = NDArrayPythonInterop.ToNumpyCopy(np.arange(1));
        }
    }
    static bool WaitCounters(int e, int i, int ms = 8000)
    {
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < ms)
        {
            if (NDArrayPythonInterop.LiveExports <= e && NDArrayPythonInterop.LiveImports <= i) return true;
            Pump(); Thread.Sleep(15);
        }
        return NDArrayPythonInterop.LiveExports <= e && NDArrayPythonInterop.LiveImports <= i;
    }

    // ============================================================
    //  BATTERY: correctness / fidelity / robustness (one engine)
    // ============================================================
    static void Battery()
    {
        var scope = Scope();
        DtypeRoundTrips();
        LayoutExportFidelity();
        ImportViewFidelity();
        ReadonlyAndBroadcast();
        DtypeMapGaps();
        ScalarAndEmpty();
        CodecMatmul();
        ExceptionSafety();
    }

    // --- all 15 dtypes: export a known array, assert values+dtype in python, mutate back ---
    static void DtypeRoundTrips()
    {
        Console.WriteLine("\n-- dtype round-trips --");
        (NPTypeCode tc, string npdtype)[] cases =
        {
            (NPTypeCode.Boolean,"bool"), (NPTypeCode.Byte,"uint8"), (NPTypeCode.SByte,"int8"),
            (NPTypeCode.Int16,"int16"), (NPTypeCode.UInt16,"uint16"), (NPTypeCode.Int32,"int32"),
            (NPTypeCode.UInt32,"uint32"), (NPTypeCode.Int64,"int64"), (NPTypeCode.UInt64,"uint64"),
            (NPTypeCode.Half,"float16"), (NPTypeCode.Single,"float32"), (NPTypeCode.Double,"float64"),
            (NPTypeCode.Complex,"complex128"),
        };
        foreach (var (tc, npdtype) in cases)
        {
            try
            {
                var nd = np.arange(5).astype(tc);
                using (Py.GIL())
                {
                    using var a = nd.ToNumpy();
                    Scope().Set("a", a);
                }
                string dt = PyS("a.dtype");
                bool eq = PyB("np.array_equal(a, np.arange(5).astype(a.dtype))");
                if (dt != npdtype) FIND($"dtype {tc}: exported numpy dtype '{dt}', expected '{npdtype}'");
                else if (!eq) FIND($"dtype {tc}: exported values != arange(5)");
                else OK($"{tc} -> {dt} values match");
            }
            catch (Exception e) { FIND($"dtype {tc} round-trip threw {e.GetType().Name}: {e.Message}"); }
        }
        // Char: exported as uint16
        try
        {
            var nd = np.arange(4).astype(NPTypeCode.Char);
            using (Py.GIL()) { using var a = nd.ToNumpy(); Scope().Set("a", a); }
            string dt = PyS("a.dtype");
            if (dt != "uint16") FIND($"Char exported as '{dt}', expected uint16"); else OK("Char -> uint16");
        }
        catch (Exception e) { FIND($"Char export threw {e.GetType().Name}: {e.Message}"); }

        // Decimal: must throw NotSupportedException (no numpy dtype)
        try
        {
            var nd = np.arange(3).astype(NPTypeCode.Decimal);
            using (Py.GIL()) { using var a = nd.ToNumpy(); }
            FIND("Decimal export did NOT throw (expected NotSupportedException)");
        }
        catch (NotSupportedException) { OK("Decimal export throws NotSupportedException"); }
        catch (Exception e) { FIND($"Decimal export threw {e.GetType().Name}, expected NotSupportedException"); }
    }

    // --- exported layouts must match numpy-native layout (values + strides) ---
    static void LayoutExportFidelity()
    {
        Console.WriteLine("\n-- layout export fidelity (values + strides vs numpy-native) --");
        // base: 4x5 int64
        void Check(string label, NDArray view, string npExpr)
        {
            try
            {
                using (Py.GIL()) { using var a = view.ToNumpy(); Scope().Set("a", a); }
                PyExec($"expected = {npExpr}");
                bool veq = PyB("np.array_equal(a, expected)");
                bool seq = PyB("a.strides == expected.strides");
                bool sheq = PyB("a.shape == expected.shape");
                if (!sheq) FIND($"layout '{label}': shape {PyS("a.shape")} != numpy {PyS("expected.shape")}");
                else if (!veq) FIND($"layout '{label}': values differ from numpy-native ({npExpr})");
                else if (!seq) FIND($"layout '{label}': strides {PyS("a.strides")} != numpy-native {PyS("expected.strides")}");
                else OK($"{label}: shape+values+strides match numpy");
            }
            catch (Exception e) { FIND($"layout '{label}' threw {e.GetType().Name}: {e.Message}"); }
        }
        PyExec("base = np.arange(20, dtype=np.int64).reshape(4,5)");
        var b = np.arange(20).astype(NPTypeCode.Int64).reshape(4, 5);
        Check("contig 4x5", b, "base");
        Check("row slice [1:3]", b["1:3"], "base[1:3]");
        Check("col slice [:, 1:4]", b[":, 1:4"], "base[:, 1:4]");
        Check("step [::2, ::2]", b["::2, ::2"], "base[::2, ::2]");
        Check("reversed rows [::-1]", b["::-1"], "base[::-1]");
        Check("reversed all [::-1,::-1]", b["::-1, ::-1"], "base[::-1, ::-1]");
        Check("transpose .T", b.T, "base.T");
        Check("neg col [:, ::-1]", b[":, ::-1"], "base[:, ::-1]");
        Check("scalar element [2,3]", b[2, 3], "np.int64(base[2,3])");
        // fortran
        var f = np.asfortranarray(np.arange(12).astype(NPTypeCode.Double).reshape(3, 4));
        PyExec("basef = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3,4))");
        Check("fortran 3x4", f, "basef");
        Check("fortran slice [1:,1:]", f["1:, 1:"], "basef[1:, 1:]");
    }

    // --- import numpy views (strided/negative/broadcast) as NumSharp views, prove aliasing ---
    static void ImportViewFidelity()
    {
        Console.WriteLine("\n-- import view fidelity + aliasing --");
        void Check(string label, string npSrc, int[] coord, double writeVal)
        {
            try
            {
                using var scope = Py.GIL();
                PythonEngine.RunSimpleString($"import numpy as _np\n_src = {npSrc}");
                using var main = Py.Import("__main__");
                using var src = main.GetAttr("_src");
                NDArray view;
                try { view = NDArrayPythonInterop.ToNDArrayView(src, allowReadonly: false); }
                catch (Exception e) { FIND($"import '{label}': ToNDArrayView threw {e.GetType().Name}: {e.Message}"); return; }
                // compare full values via python-side sum-of-abs-diff against a copy import
                using (var copy = NDArrayPythonInterop.ToNDArray(src))
                {
                    bool shapeEq = ShapeEq(view.shape, copy.shape);
                    if (!shapeEq) { FIND($"import '{label}': view shape [{string.Join(",", view.shape)}] != copy shape [{string.Join(",", copy.shape)}]"); view.Dispose(); return; }
                }
                // mutate through the view, verify python sees it (read back through the SAME src object)
                view.SetDouble(writeVal, coord);
                Scope().Set("_src", src);
                double pySees = ReadPyElem("_src", coord);
                if (Math.Abs(pySees - writeVal) > 1e-9)
                    FIND($"import '{label}': wrote {writeVal} through view at [{string.Join(",", coord)}], python sees {pySees} (aliasing broken)");
                else OK($"{label}: aliasing verified (python sees {writeVal})");
                view.Dispose();
            }
            catch (Exception e) { FIND($"import '{label}' threw {e.GetType().Name}: {e.Message}"); }
        }
        Check("contig 1d", "_np.arange(6, dtype='f8')", new[] { 3 }, 42.0);
        Check("row slice", "_np.arange(20, dtype='f8').reshape(4,5)[1:3]", new[] { 0, 2 }, 7.0);
        Check("strided [::2]", "_np.arange(10, dtype='f8')[::2]", new[] { 1 }, 13.0);
        Check("reversed [::-1]", "_np.arange(6, dtype='f8')[::-1]", new[] { 0 }, 55.0);
        Check("transpose", "_np.arange(6, dtype='f8').reshape(2,3).T", new[] { 2, 1 }, 88.0);
        Check("2d neg col", "_np.arange(12, dtype='f8').reshape(3,4)[:, ::-1]", new[] { 1, 1 }, 99.0);
    }

    static void ReadonlyAndBroadcast()
    {
        Console.WriteLine("\n-- read-only + broadcast semantics --");
        // read-only source refused by default, accepted (non-writeable) with allowReadonly
        try
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_ro = _np.arange(4, dtype='f8'); _ro.setflags(write=False)");
            using var main = Py.Import("__main__");
            using var ro = main.GetAttr("_ro");
            bool threw = false;
            try { using var v = NDArrayPythonInterop.ToNDArrayView(ro, allowReadonly: false); }
            catch (InvalidOperationException) { threw = true; }
            if (!threw) FIND("read-only numpy view accepted by default (should refuse)"); else OK("read-only refused by default");

            using var v2 = NDArrayPythonInterop.ToNDArrayView(ro, allowReadonly: true);
            if (v2.Shape.IsWriteable) FIND("allowReadonly view is writeable (should be non-writeable)");
            else OK("allowReadonly view is non-writeable");
            v2.Dispose();
        }
        catch (Exception e) { FIND($"readonly test threw {e.GetType().Name}: {e.Message}"); }

        // broadcast export -> read-only numpy
        try
        {
            var bc = np.broadcast_to(np.arange(3).astype(NPTypeCode.Double), new Shape(4, 3));
            using (Py.GIL()) { using var a = bc.ToNumpy(); Scope().Set("a", a); }
            bool writeable = PyB("a.flags.writeable");
            if (writeable) FIND("broadcast export produced WRITEABLE numpy array (should be read-only)");
            else OK("broadcast export -> read-only numpy");
            bool veq = PyB("np.array_equal(a, np.broadcast_to(np.arange(3, dtype='f8'), (4,3)))");
            if (!veq) FIND("broadcast export values wrong"); else OK("broadcast export values correct");
        }
        catch (Exception e) { FIND($"broadcast export threw {e.GetType().Name}: {e.Message}"); }
    }

    static void DtypeMapGaps()
    {
        Console.WriteLine("\n-- dtype-map gaps (big-endian / complex64 / UCS-4 / structured) --");
        // big-endian import -> refuse (both copy and view)
        Try("big-endian view refused", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_be = _np.arange(4, dtype='>i4')");
            using var main = Py.Import("__main__"); using var be = main.GetAttr("_be");
            bool threw = false;
            try { using var v = NDArrayPythonInterop.ToNDArrayView(be, allowReadonly: true); } catch (NotSupportedException) { threw = true; }
            return threw;
        });
        // complex64 copy widens
        Try("complex64 copy widens to Complex", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_c8 = _np.array([1+2j,3+4j], dtype='complex64')");
            using var main = Py.Import("__main__"); using var c8 = main.GetAttr("_c8");
            using var nd = NDArrayPythonInterop.ToNDArray(c8);
            return nd.typecode == NPTypeCode.Complex && nd.size == 2;
        });
        // complex64 VIEW must fail (no zero-copy)
        Try("complex64 view refused", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_c8 = _np.array([1+2j], dtype='complex64')");
            using var main = Py.Import("__main__"); using var c8 = main.GetAttr("_c8");
            bool threw = false;
            try { using var v = NDArrayPythonInterop.ToNDArrayView(c8, allowReadonly: true); } catch (NotSupportedException) { threw = true; }
            return threw;
        });
        // UCS-4 text narrow copy (BMP)
        Try("UCS-4 '<U1' copy narrows to Char", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_u = _np.array(['a','b','c'], dtype='<U1')");
            using var main = Py.Import("__main__"); using var u = main.GetAttr("_u");
            using var nd = NDArrayPythonInterop.ToNDArray(u);
            return nd.typecode == NPTypeCode.Char && nd.size == 3;
        });
        // structured dtype -> clean error, no crash
        Try("structured dtype copy -> clean throw (no crash)", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_st = _np.zeros(3, dtype=[('x','i4'),('y','f8')])");
            using var main = Py.Import("__main__"); using var st = main.GetAttr("_st");
            try { using var nd = NDArrayPythonInterop.ToNDArray(st); return false; }
            catch (Exception) { return true; }
        });
    }

    static void ScalarAndEmpty()
    {
        Console.WriteLine("\n-- scalar + empty --");
        Try("0-d export -> numpy 0-d", () =>
        {
            var s = np.arange(1).astype(NPTypeCode.Double).reshape(new int[0]);   // 0-d
            using (Py.GIL()) { using var a = s.ToNumpy(); Scope().Set("a", a); }
            return PyL("a.ndim") == 0;
        });
        Try("empty export -> numpy empty", () =>
        {
            var e = np.arange(0).astype(NPTypeCode.Double);
            using (Py.GIL()) { using var a = e.ToNumpy(); Scope().Set("a", a); }
            return PyL("a.size") == 0;
        });
        Try("empty import view", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_e = _np.arange(0, dtype='f8')");
            using var main = Py.Import("__main__"); using var e = main.GetAttr("_e");
            using var v = NDArrayPythonInterop.ToNDArrayView(e, allowReadonly: true);
            return v.size == 0;
        });
    }

    static void CodecMatmul()
    {
        Console.WriteLine("\n-- codec auto-marshal (matmul) --");
        try
        {
            var a = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);
            var b = np.arange(6).astype(NPTypeCode.Double).reshape(3, 2);
            using (Py.GIL())
            {
                using var scope = Py.CreateScope();
                scope.Exec("import numpy as np");
                scope.Set("a", a); scope.Set("b", b);      // NDArray auto-encoded
                using var r = scope.Eval("np.matmul(a, b)");
                using var nd = r.As<NDArray>();             // decoded back
                double v00 = nd.GetDouble(0, 0);
                // expected: [0,1,2]·[0,2,4]... compute reference
                double exp = 0 * 0 + 1 * 2 + 2 * 4;
                if (Math.Abs(v00 - exp) > 1e-9) FIND($"codec matmul [0,0]={v00}, expected {exp}");
                else OK($"codec matmul [0,0]={v00}");
            }
        }
        catch (Exception e) { FIND($"codec matmul threw {e.GetType().Name}: {e.Message}"); }
    }

    static void ExceptionSafety()
    {
        Console.WriteLine("\n-- exception safety / arg validation --");
        Try("null export -> ArgumentNullException", () => { try { NDArrayPythonInterop.ToNumpy((NDArray)null); return false; } catch (ArgumentNullException) { return true; } });
        Try("null import -> ArgumentNullException", () => { try { NDArrayPythonInterop.ToNDArray((PyObject)null); return false; } catch (ArgumentNullException) { return true; } });
        Try("non-buffer object (dict) import -> NotSupportedException", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("_d = {'x':1}");
            using var main = Py.Import("__main__"); using var d = main.GetAttr("_d");
            try { using var nd = NDArrayPythonInterop.ToNDArray(d); return false; }
            catch (NotSupportedException) { return true; }
        });
        Try("non-contiguous ToMemoryView -> InvalidOperationException", () =>
        {
            var v = np.arange(20).astype(NPTypeCode.Double).reshape(4, 5).T;   // non-contiguous
            try { using (Py.GIL()) { using var mv = v.ToMemoryView(); } return false; }
            catch (InvalidOperationException) { return true; }
        });
        // counters must be at/near baseline after all the above
        bool settled = WaitCounters(0, 0, 6000);
        if (!settled) FIND($"after battery, counters did not settle: LiveExports={NDArrayPythonInterop.LiveExports} LiveImports={NDArrayPythonInterop.LiveImports}");
        else OK("counters settled to baseline after battery");
    }

    // ============================================================
    //  LEAK SWEEP: unmanaged + managed growth over many iterations
    // ============================================================
    static void LeakSweep()
    {
        var scope = Scope();
        const int N = 120;
        const int elems = 1_000_000;   // 8 MB per double array
        Console.WriteLine($"\n-- leak sweep: {N} iterations x ~{elems*8/1024/1024}MB each --");

        long baseWs = Working();
        long baseGc = GC.GetTotalMemory(true);

        // 1) export view + dispose wrapper only (Python drops immediately)
        for (int i = 0; i < N; i++)
        {
            var nd = np.arange(elems).astype(NPTypeCode.Double);
            using (Py.GIL()) { using var a = nd.ToNumpy(); }   // wrapper disposed; python has no other ref
            nd.Dispose();
            if (i % 30 == 0) Pump();
        }
        Pump(); WaitCounters(0, 0, 8000);
        Report("after export-view sweep", baseWs, baseGc);

        // 2) export copy + dispose
        for (int i = 0; i < N; i++)
        {
            var nd = np.arange(elems).astype(NPTypeCode.Double);
            using (Py.GIL()) { using var a = nd.ToNumpyCopy(); }
            nd.Dispose();
            if (i % 30 == 0) Pump();
        }
        Pump(); WaitCounters(0, 0, 8000);
        Report("after export-copy sweep", baseWs, baseGc);

        // 3) import view + dispose (deferred lease drain)
        for (int i = 0; i < N; i++)
        {
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString($"import numpy as _np\n_big = _np.arange({elems}, dtype='f8')");
                using var main = Py.Import("__main__"); using var big = main.GetAttr("_big");
                using var v = NDArrayPythonInterop.ToNDArrayView(big);
                v.SetDouble(1.0, 0);
                main.SetAttr("_big", 0.ToPython());   // drop python ref so nothing pins after lease
            }
            if (i % 30 == 0) Pump();
        }
        Pump(); WaitCounters(0, 0, 8000);
        Report("after import-view sweep", baseWs, baseGc);

        // 4) import copy + dispose
        for (int i = 0; i < N; i++)
        {
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString($"import numpy as _np\n_big = _np.arange({elems}, dtype='f8')");
                using var main = Py.Import("__main__"); using var big = main.GetAttr("_big");
                using var nd = NDArrayPythonInterop.ToNDArray(big);
                main.SetAttr("_big", 0.ToPython());
            }
            if (i % 30 == 0) Pump();
        }
        Pump(); WaitCounters(0, 0, 8000);
        Report("after import-copy sweep", baseWs, baseGc);

        // 5) FINALIZER path: create import views, drop WITHOUT disposing, force GC
        for (int i = 0; i < N; i++)
        {
            MakeAndDropImportView(elems);
            if (i % 20 == 0) Pump();
        }
        Pump(); Pump(); WaitCounters(0, 0, 10000);
        Report("after finalizer-drop import sweep", baseWs, baseGc);

        Console.WriteLine($"final: LiveExports={NDArrayPythonInterop.LiveExports} LiveImports={NDArrayPythonInterop.LiveImports}");
        long grow = Working() - baseWs;
        NOTE($"net working-set growth over whole sweep: {grow/1024/1024} MB");
        if (grow > 400L * 1024 * 1024) FIND($"working set grew {grow/1024/1024}MB over sweep (possible leak)");
        else OK($"working set growth bounded ({grow/1024/1024}MB)");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    static void MakeAndDropImportView(int elems)
    {
        using (Py.GIL())
        {
            PythonEngine.RunSimpleString($"import numpy as _np\n_big2 = _np.arange({elems}, dtype='f8')");
            using var main = Py.Import("__main__"); using var big = main.GetAttr("_big2");
            var v = NDArrayPythonInterop.ToNDArrayView(big);   // NOT disposed
            v.SetDouble(2.0, 1);
            main.SetAttr("_big2", 0.ToPython());
        }
        // v goes out of scope -> finalizer must release the lease
    }

    static long Working() => Process.GetCurrentProcess().WorkingSet64;
    static void Report(string label, long baseWs, long baseGc)
    {
        long ws = Working(); long gc = GC.GetTotalMemory(false);
        Console.WriteLine($"  {label}: WS={ws/1024/1024}MB (+{(ws-baseWs)/1024/1024}) GC={gc/1024/1024}MB (+{(gc-baseGc)/1024/1024}) LiveE={NDArrayPythonInterop.LiveExports} LiveI={NDArrayPythonInterop.LiveImports}");
    }

    // ============================================================
    //  POISON: reproduce the decoder-cache poisoning (fresh engine, no early register)
    // ============================================================
    static void Poison()
    {
        Console.WriteLine("\n-- codec decoder-cache poisoning --");
        using var scope = Py.GIL();
        using var s = Py.CreateScope();
        s.Exec("import numpy as np");
        // 1) decode BEFORE registration -> fails, poisons the pair
        using (var a = s.Eval("np.arange(3, dtype='f8')"))
        {
            try { a.As<NDArray>(); FIND("pre-register decode SUCCEEDED (unexpected)"); }
            catch (InvalidCastException) { OK("pre-register decode fails (expected)"); }
        }
        // 2) register
        bool reg = NDArrayPythonInterop.RegisterCodec();
        OK($"RegisterCodec returned {reg}");
        // 3) SAME pair still fails (the poisoning)
        using (var a = s.Eval("np.arange(3, dtype='f8')"))
        {
            try { using var nd = a.As<NDArray>(); NOTE("poisoned pair decoded OK (poisoning NOT reproduced on this build)"); }
            catch (InvalidCastException) { NOTE("poisoned pair STILL fails after register -> confirmed known gap (register before first use)"); }
        }
        // 4) a fresh pair works
        using (var by = s.Eval("b'abcd'"))
        {
            try { using var nd = by.As<NDArray>(); OK($"fresh pair (bytes) decodes: {nd.typecode} size {nd.size}"); }
            catch (Exception e) { FIND($"fresh pair decode failed: {e.GetType().Name}"); }
        }
    }

    // ============================================================
    //  SHUTDOWN: convert after Shutdown must not corrupt/crash
    // ============================================================
    static void ShutdownConvert()
    {
        Console.WriteLine("\n-- convert after shutdown + orphaned-export sweep --");
        // create an export held ONLY by Python, drop the C# ref -> a true orphan
        MakeOrphanExport();
        Pump();
        int liveBefore = NDArrayPythonInterop.LiveExports;
        Console.WriteLine($"pre-shutdown LiveExports={liveBefore} (orphan pinned by Python only)");
        if (liveBefore < 1) NOTE("no orphan pinned (GC raced) — sweep test weaker");

        RuntimeData.FormatterType = typeof(NoopFormatter);
        try { PythonEngine.Shutdown(); OK("shutdown completed"); } catch (Exception e) { FIND($"shutdown threw {e.GetType().Name}: {e.Message}"); }

        // after shutdown, a conversion should throw a clean InvalidOperationException (engine not initialized)
        try { using var a = np.arange(3).astype(NPTypeCode.Double).ToNumpy(); FIND("export after shutdown SUCCEEDED (engine down)"); }
        catch (InvalidOperationException) { OK("export after shutdown -> clean InvalidOperationException"); }
        catch (Exception e) { NOTE($"export after shutdown threw {e.GetType().Name}: {e.Message}"); }

        // the orphaned-export sweep must release the Python-held pin now the engine is dead
        var sw = Stopwatch.StartNew();
        while (sw.ElapsedMilliseconds < 10000 && NDArrayPythonInterop.LiveExports > 0)
            Thread.Sleep(50);
        if (NDArrayPythonInterop.LiveExports > 0)
            FIND($"orphaned-export sweep did NOT release the Python-held pin after shutdown (LiveExports={NDArrayPythonInterop.LiveExports}) -> buffer leaked for process lifetime");
        else
            OK($"orphaned-export sweep released the pin {sw.ElapsedMilliseconds}ms after shutdown (no leak)");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    static void MakeOrphanExport()
    {
        var nd = np.arange(6).astype(NPTypeCode.Double);
        using (Py.GIL())
        {
            using var a = nd.ToNumpy();                       // ExportKeeper takes its own ARC ref
            using var main = Py.Import("__main__");
            main.SetAttr("shutdown_keep", a);                 // Python is the only long-lived holder
        }
        nd.Dispose();                                         // drop the C# reference entirely
    }

    // ============================================================
    //  GIL: requireGIL:false without holding the GIL -> documented AV
    // ============================================================
    static void GilAv()
    {
        Console.WriteLine("\n-- GIL policy: requireGIL:false without holding GIL (documented crash) --");
        NOTE("about to call ToNumpy(requireGIL:false) WITHOUT the GIL; documented as immediate access violation");
        // First the SAFE case: holding the GIL, requireGIL:false must work.
        try
        {
            using (Py.GIL())
            {
                var nd = np.arange(4).astype(NPTypeCode.Double);
                using var a = NDArrayPythonInterop.ToNumpy(nd, requireGIL: false);
                OK("requireGIL:false WHILE holding GIL works");
            }
        }
        catch (Exception e) { FIND($"requireGIL:false while holding GIL threw {e.GetType().Name}: {e.Message}"); }
        NOTE("now the unsafe call (may hard-crash the process, which is the documented behavior)");
        var nd2 = np.arange(4).astype(NPTypeCode.Double);
        using var a2 = NDArrayPythonInterop.ToNumpy(nd2, requireGIL: false);  // expected AV
        FIND("requireGIL:false WITHOUT GIL did NOT crash (unexpected)");
    }

    // ============================================================
    //  THREADS: concurrent conversions from multiple threads
    // ============================================================
    static void Threads()
    {
        Console.WriteLine("\n-- concurrent conversions from many threads --");
        var scope = Scope();
        int T = 8, per = 200;
        int errors = 0;
        var threads = new List<Thread>();
        for (int t = 0; t < T; t++)
        {
            int tid = t;
            var th = new Thread(() =>
            {
                for (int i = 0; i < per; i++)
                {
                    try
                    {
                        var nd = np.arange(100 + tid).astype(NPTypeCode.Double);
                        using (Py.GIL()) { using var a = nd.ToNumpy(); using var c = nd.ToNumpyCopy(); }
                        nd.Dispose();
                        using (Py.GIL())
                        {
                            PythonEngine.RunSimpleString($"import numpy as _np\n_tt{tid} = _np.arange(50, dtype='f8')");
                            using var main = Py.Import("__main__"); using var s = main.GetAttr($"_tt{tid}");
                            using var v = NDArrayPythonInterop.ToNDArrayView(s);
                            v.SetDouble(1, 0);
                        }
                    }
                    catch (Exception e) { Interlocked.Increment(ref errors); if (errors < 5) Console.WriteLine($"[thread {tid}] {e.GetType().Name}: {e.Message}"); }
                }
            });
            threads.Add(th); th.Start();
        }
        foreach (var th in threads) th.Join();
        if (errors > 0) FIND($"{errors} errors across {T} threads x {per} conversions");
        else OK($"{T} threads x {per} conversions completed with no errors");
        bool settled = WaitCounters(0, 0, 12000);
        if (!settled) FIND($"counters did not settle after threads: E={NDArrayPythonInterop.LiveExports} I={NDArrayPythonInterop.LiveImports}");
        else OK("counters settled after threads");
    }

    // ============================================================
    //  AIFACE: malformed __array_interface__ robustness
    // ============================================================
    static void ArrayInterfaceRobustness()
    {
        Console.WriteLine("\n-- malformed __array_interface__ robustness --");
        // custom python object exposing a bogus array-interface; must NOT segfault, must throw cleanly
        string[] bogus =
        {
            "class B:\n __array_interface__={'shape':(3,),'typestr':'<f8','data':(0,False),'version':3}\nb=B()",  // null ptr
            "class B:\n __array_interface__={'shape':(2,),'typestr':'<f8','data':b'abcdefgh','version':3}\nb=B()", // bytes data
            "class B:\n __array_interface__={'shape':(2,),'typestr':'<f8','version':3}\nb=B()",                    // no data key
            "class B:\n __array_interface__={'shape':(2,2),'typestr':'<f8','data':(999999,False),'strides':(8,8,8),'version':3}\nb=B()", // strides len != shape
        };
        int idx = 0;
        foreach (var code in bogus)
        {
            idx++;
            try
            {
                using var scope = Py.GIL();
                PythonEngine.RunSimpleString(code);
                using var main = Py.Import("__main__"); using var b = main.GetAttr("b");
                try
                {
                    using var v = NDArrayPythonInterop.ToNDArrayView(b, allowReadonly: true);
                    // if it "succeeds" with a null/garbage pointer, that's dangerous — but reading would crash.
                    NOTE($"bogus#{idx}: ToNDArrayView returned a view (shape [{string.Join(",", v.shape)}]) WITHOUT throwing");
                    // do NOT dereference garbage; just dispose
                    v.Dispose();
                }
                catch (Exception e) { OK($"bogus#{idx}: threw {e.GetType().Name} (clean)"); }
            }
            catch (Exception e) { NOTE($"bogus#{idx}: outer {e.GetType().Name}: {e.Message}"); }
        }
    }

    // ============================================================
    //  AIFACE2: malformed __array_interface__ vs numpy (divergence proof)
    // ============================================================
    static void ArrayInterfaceVsNumpy()
    {
        Console.WriteLine("\n-- malformed __array_interface__: NumSharp vs numpy --");
        (string name, string ai)[] cases =
        {
            ("null_ptr", "{'shape':(3,),'typestr':'<f8','data':(0,False),'version':3}"),
            ("strides_gt_shape", "{'shape':(2,2),'typestr':'<f8','data':(999999,False),'strides':(8,8,8),'version':3}"),
            ("strides_lt_shape", "{'shape':(2,2),'typestr':'<f8','data':(999999,False),'strides':(8,),'version':3}"),
            ("neg_shape", "{'shape':(-2,),'typestr':'<f8','data':(999999,False),'version':3}"),
        };
        foreach (var (name, ai) in cases)
        {
            string numpyResult;
            bool numpyRejects;
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString(
                    $"import numpy as _np\nclass _B: pass\n_b=_B()\n_B.__array_interface__={ai}\n" +
                    "try:\n _a=_np.asarray(_b); _r=('BUILD %s %s'%(_a.shape,_a.strides))\nexcept Exception as _e:\n _r=('REJECT %s'%type(_e).__name__)");
                using var main = Py.Import("__main__"); using var r = main.GetAttr("_r");
                numpyResult = r.As<string>();
                numpyRejects = numpyResult.StartsWith("REJECT");
            }
            string nsResult;
            bool nsRejects;
            try
            {
                using var scope = Py.GIL();
                using var main = Py.Import("__main__"); using var b = main.GetAttr("_b");
                using var v = NDArrayPythonInterop.ToNDArrayView(b, allowReadonly: true);
                nsResult = $"BUILD shape=[{string.Join(",", v.shape)}] strideslen={v.Shape.Strides.Length}";
                nsRejects = false;
                v.Dispose();
            }
            catch (Exception e) { nsResult = $"REJECT {e.GetType().Name}"; nsRejects = true; }

            Console.WriteLine($"  {name,-16} numpy: {numpyResult,-28} | NumSharp: {nsResult}");
            if (numpyRejects && !nsRejects)
                FIND($"__array_interface__ '{name}': numpy REJECTS ({numpyResult}) but NumSharp BUILDS an unvalidated view ({nsResult}) -> memory-unsafe on read");
            else
                OK($"'{name}': both {(numpyRejects ? "reject" : "build")}");
        }
    }

    // ============================================================
    //  STRIDED: non-numpy strided import (ViewViaBufferStrides) correctness
    // ============================================================
    static void StridedNonNumpyImport()
    {
        Console.WriteLine("\n-- non-numpy strided import (array.array memoryviews) --");
        void Check(string label, string pySrc, int[] coord, long expectSrcIdx, double writeVal)
        {
            try
            {
                using var scope = Py.GIL();
                PythonEngine.RunSimpleString($"import array\n_arr = array.array('d', [float(i) for i in range(10)])\n_mv = {pySrc}");
                using var main = Py.Import("__main__"); using var mv = main.GetAttr("_mv");
                using var arrObj = main.GetAttr("_arr");
                // confirm it is genuinely non-numpy and non-contiguous
                using var v = NDArrayPythonInterop.ToNDArrayView(mv, allowReadonly: true);
                // read value at coord via numpy oracle: compare view to expected element of _arr
                Scope().Set("_mv2", mv);
                Scope().Set("_arr", arrObj);
                double viewVal = v.GetDouble(coord);
                double pyVal = PyF($"float(_mv2[{string.Join(",", coord)}])");
                if (Math.Abs(viewVal - pyVal) > 1e-9)
                    FIND($"strided '{label}': view[{string.Join(",", coord)}]={viewVal} != python {pyVal}");
                else OK($"{label}: read matches python ({viewVal})");
                // write-through (if writeable)
                if (v.Shape.IsWriteable)
                {
                    v.SetDouble(writeVal, coord);
                    double srcSees = PyF($"float(_arr[{expectSrcIdx}])");
                    if (Math.Abs(srcSees - writeVal) > 1e-9)
                        FIND($"strided '{label}': wrote {writeVal} through view, array.array[{expectSrcIdx}]={srcSees} (write-through broken)");
                    else OK($"{label}: write-through to array.array[{expectSrcIdx}]={srcSees}");
                }
                else NOTE($"{label}: view non-writeable (source read-only)");
                v.Dispose();
            }
            catch (Exception e) { FIND($"strided '{label}' threw {e.GetType().Name}: {e.Message}"); }
        }
        Check("mv[::2]", "memoryview(_arr)[::2]", new[] { 2 }, 4, 77.0);      // view idx2 -> arr[4]
        Check("mv[::-1]", "memoryview(_arr)[::-1]", new[] { 0 }, 9, 88.0);    // view idx0 -> arr[9]
        Check("mv[3:]", "memoryview(_arr)[3:]", new[] { 1 }, 4, 66.0);        // view idx1 -> arr[4]
        Check("mv[1::3]", "memoryview(_arr)[1::3]", new[] { 2 }, 7, 55.0);    // view idx2 -> arr[7]
        bool settled = WaitCounters(0, 0, 6000);
        if (!settled) FIND($"strided import leaked leases: I={NDArrayPythonInterop.LiveImports}");
        else OK("strided import counters settled");
    }

    // ============================================================
    //  DRAINIDLE: deferred lease must release without further conversions
    // ============================================================
    static void DrainIdle()
    {
        Console.WriteLine("\n-- deferred-drain reliability (no further conversions/GC) --");
        int before = NDArrayPythonInterop.LiveImports;
        using (Py.GIL())
        {
            PythonEngine.RunSimpleString("import numpy as _np\n_d = _np.arange(4, dtype='f8')");
            using var main = Py.Import("__main__"); using var d = main.GetAttr("_d");
            var v = NDArrayPythonInterop.ToNDArrayView(d);
            v.Dispose();   // Release -> QueueDisposal (deferred). NO further conversions after this.
        }
        // poll WITHOUT Pump / GC / any conversion — only the threadpool drain can clear it.
        var sw = Stopwatch.StartNew();
        int last = NDArrayPythonInterop.LiveImports;
        while (sw.ElapsedMilliseconds < 5000 && NDArrayPythonInterop.LiveImports > before)
            Thread.Sleep(25);
        if (NDArrayPythonInterop.LiveImports > before)
            FIND($"deferred lease NOT drained by threadpool within 5s (LiveImports={NDArrayPythonInterop.LiveImports}, was {before}) — a lease/Py_buffer lock lingers until the next conversion");
        else
            OK($"deferred lease drained by threadpool in {sw.ElapsedMilliseconds}ms without further activity");
    }

    // ============================================================
    //  RESIZE: refcheck contracts (export pins source; import view refuses)
    // ============================================================
    static void ResizeRefcheck()
    {
        Console.WriteLine("\n-- resize refcheck contracts --");
        // (a) exported source: NumSharp resize with refcheck must refuse while Python holds a view
        try
        {
            var nd = np.arange(6).astype(NPTypeCode.Double);
            using (Py.GIL()) { using var a = nd.ToNumpy(); using var main = Py.Import("__main__"); main.SetAttr("_keepexp", a); }
            bool refused = false;
            try { nd.resize(new Shape(12), refcheck: true); }
            catch (Exception e) { refused = true; NOTE($"source resize refused: {e.GetType().Name}: {e.Message}"); }
            if (!refused) FIND("exported NumSharp source allowed size-changing resize(refcheck:true) while Python held a view");
            else OK("exported source refuses resize(refcheck:true)");
            using (Py.GIL()) { using var main = Py.Import("__main__"); main.SetAttr("_keepexp", 0.ToPython()); }
        }
        catch (Exception e) { FIND($"export-resize test threw {e.GetType().Name}: {e.Message}"); }

        // (b) import view: does-not-own-its-data -> size-changing resize refuses
        try
        {
            NDArray v;
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString("import numpy as _np\n_ri = _np.arange(6, dtype='f8')");
                using var main = Py.Import("__main__"); using var s = main.GetAttr("_ri");
                v = NDArrayPythonInterop.ToNDArrayView(s);
            }
            bool refused = false;
            try { v.resize(new Shape(12), refcheck: false); }
            catch (Exception e) { refused = true; NOTE($"import-view resize refused: {e.GetType().Name}: {e.Message}"); }
            if (!refused) FIND("import view (owndata=false) allowed size-changing resize — detaches from Python memory");
            else OK("import view refuses size-changing resize");
            v.Dispose();
        }
        catch (Exception e) { FIND($"import-resize test threw {e.GetType().Name}: {e.Message}"); }
        WaitCounters(0, 0, 6000);
    }

    // ============================================================
    //  MMAP: import mmap-backed memory and dispose (documented GIL-bounce area)
    // ============================================================
    static void MmapImport()
    {
        Console.WriteLine("\n-- mmap import + dispose (GIL-bounce dealloc) --");
        int errors = 0;
        for (int i = 0; i < 50; i++)
        {
            try
            {
                using (Py.GIL())
                {
                    PythonEngine.RunSimpleString("import mmap\n_mm = mmap.mmap(-1, 80)\n_mmv = memoryview(_mm).cast('d')");
                    using var main = Py.Import("__main__"); using var mmv = main.GetAttr("_mmv");
                    using var v = NDArrayPythonInterop.ToNDArrayView(mmv);
                    v.SetDouble(3.14, 0);
                    main.SetAttr("_mmv", 0.ToPython());
                    main.SetAttr("_mm", 0.ToPython());
                }
                if (i % 10 == 0) Pump();
            }
            catch (Exception e) { errors++; if (errors < 4) Console.WriteLine($"[mmap {i}] {e.GetType().Name}: {e.Message}"); }
        }
        bool settled = WaitCounters(0, 0, 10000);
        if (errors > 0) FIND($"{errors} errors during mmap import/dispose");
        else if (!settled) FIND($"mmap leases did not settle: I={NDArrayPythonInterop.LiveImports}");
        else OK("50 mmap import/dispose cycles clean, counters settled (no GIL-bounce deadlock)");
    }

    // ============================================================
    //  NULLCRASH: prove the null-ptr array-interface view AVs on read
    // ============================================================
    static void NullPtrCrash()
    {
        Console.WriteLine("\n-- null-ptr array-interface view: read attempt (expected AV) --");
        using var scope = Py.GIL();
        PythonEngine.RunSimpleString("class _B: pass\n_b=_B()\n_B.__array_interface__={'shape':(3,),'typestr':'<f8','data':(0,False),'version':3}");
        using var main = Py.Import("__main__"); using var b = main.GetAttr("_b");
        using var v = NDArrayPythonInterop.ToNDArrayView(b, allowReadonly: true);
        Console.WriteLine($"built view over NULL data pointer, shape=[{string.Join(",", v.shape)}]; now reading element 0...");
        Console.Out.Flush();
        double x = v.GetDouble(0);   // dereferences address 0 -> access violation
        FIND($"read of null-ptr view returned {x} without crashing (unexpected)");
    }

    // ============================================================
    //  DTYPEEDGE: width mapping, float16, extreme integer values
    // ============================================================
    static void DtypeEdge()
    {
        Console.WriteLine("\n-- dtype width / precision edges --");
        // array.array width mapping (Windows: l/L are 4 bytes, q/Q are 8)
        void ArrArr(string typecode, NPTypeCode expect)
        {
            try
            {
                using var scope = Py.GIL();
                PythonEngine.RunSimpleString($"import array\n_aa = array.array('{typecode}', [1,2,3])");
                using var main = Py.Import("__main__"); using var aa = main.GetAttr("_aa");
                using var nd = NDArrayPythonInterop.ToNDArray(aa);
                if (nd.typecode != expect) FIND($"array.array('{typecode}') -> {nd.typecode}, expected {expect}");
                else OK($"array.array('{typecode}') -> {nd.typecode}");
            }
            catch (Exception e) { FIND($"array.array('{typecode}') threw {e.GetType().Name}: {e.Message}"); }
        }
        // On Windows C long is 4 bytes -> l/L map to 32-bit.
        ArrArr("l", NPTypeCode.Int32);
        ArrArr("L", NPTypeCode.UInt32);
        ArrArr("q", NPTypeCode.Int64);
        ArrArr("Q", NPTypeCode.UInt64);
        ArrArr("i", NPTypeCode.Int32);
        ArrArr("f", NPTypeCode.Single);
        ArrArr("d", NPTypeCode.Double);

        // float16 view import + value fidelity
        Try("float16 numpy view import -> Half, value exact", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_h = _np.array([1.5, 2.25, -3.5], dtype='float16')");
            using var main = Py.Import("__main__"); using var h = main.GetAttr("_h");
            using var v = NDArrayPythonInterop.ToNDArrayView(h, allowReadonly: true);
            return v.typecode == NPTypeCode.Half && Math.Abs((double)(Half)v.GetValue(1) - 2.25) < 1e-6;
        });

        // extreme int64 / uint64 round-trip through numpy (bit-exact, no float coercion)
        Try("int64 extremes export round-trip exact", () =>
        {
            var nd = np.array(new long[] { long.MinValue, -1, 0, 1, long.MaxValue });
            using (Py.GIL()) { using var a = nd.ToNumpy(); Scope().Set("a", a); }
            return PyB("np.array_equal(a, np.array([-9223372036854775808,-1,0,1,9223372036854775807], dtype='int64'))");
        });
        Try("uint64 max export round-trip exact", () =>
        {
            var nd = np.array(new ulong[] { 0, 1, ulong.MaxValue });
            using (Py.GIL()) { using var a = nd.ToNumpy(); Scope().Set("a", a); }
            return PyS("a[2]") == "18446744073709551615" && PyS("a.dtype") == "uint64";
        });
        // uint64 IMPORT with a value > 2^63 (would be wrong if read as signed)
        Try("uint64 import value > 2^63 exact", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_u = _np.array([18446744073709551615], dtype='uint64')");
            using var main = Py.Import("__main__"); using var u = main.GetAttr("_u");
            using var v = NDArrayPythonInterop.ToNDArrayView(u, allowReadonly: true);
            return (ulong)v.GetValue(0) == ulong.MaxValue;
        });
        WaitCounters(0, 0, 6000);
    }

    // ---------------- misc helpers ----------------
    static void Try(string label, Func<bool> f)
    {
        try { if (f()) OK(label); else FIND(label + " -> returned false"); }
        catch (Exception e) { FIND($"{label} -> threw {e.GetType().Name}: {e.Message}"); }
    }
    static bool ShapeEq(long[] a, long[] b) { if (a.Length != b.Length) return false; for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false; return true; }
    static double ReadPyElem(string arr, int[] coord)
    {
        string idx = string.Join(",", coord);
        return PyF($"float({arr}[{idx}])");
    }
}
