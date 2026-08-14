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
                case "fuzz": Fuzz(); break;
                case "overlap": OverlapImport(); break;
                case "valuefidelity": ValueFidelity(); break;
                case "racecodec": RaceCodec(); break;
                case "getset": GetSetSemantics(); break;
                case "codecfuzz": CodecFuzz(); break;
                case "torture": Torture(); break;
                case "memviewfuzz": MemViewFuzz(); break;
                case "soak": Soak(); break;
                case "byteident": ByteIdent(); break;
                case "nansweep": NanSweep(); break;
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
    static bool PyB(string expr) { using (Py.GIL()) { using var r = Scope().Eval($"bool({expr})"); return r.As<bool>(); } }   // bool() coerces numpy.bool_ -> python bool
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

    // ============================================================
    //  FUZZ: property-based export/import fidelity vs numpy (shared slice syntax)
    // ============================================================
    static void Fuzz()
    {
        var scope = Scope();
        var rng = new Random(20260814);
        (NPTypeCode tc, string np)[] dt =
        {
            (NPTypeCode.Int32,"int32"),(NPTypeCode.Int64,"int64"),(NPTypeCode.Single,"float32"),
            (NPTypeCode.Double,"float64"),(NPTypeCode.Byte,"uint8"),(NPTypeCode.SByte,"int8"),
            (NPTypeCode.Int16,"int16"),(NPTypeCode.UInt16,"uint16"),(NPTypeCode.UInt32,"uint32"),
            (NPTypeCode.UInt64,"uint64"),(NPTypeCode.Half,"float16"),
        };
        int N = 4000, exportFails = 0, copyFails = 0, aliasFails = 0, ran = 0, skipped = 0, parseDiv = 0;
        for (int it = 0; it < N; it++)
        {
            var (tc, npn) = dt[rng.Next(dt.Length)];
            int ndim = rng.Next(1, 4);
            int[] shape = new int[ndim];
            int prod = 1;
            for (int i = 0; i < ndim; i++) { shape[i] = rng.Next(1, 6); prod *= shape[i]; }
            if (prod > 60) { skipped++; continue; }     // keep arange values exact for uint8/float16
            string shapeStr = string.Join(",", shape);
            string slice = RandomSlice(rng, shape);
            string npIndex = slice.Length == 0 ? "" : "[" + slice + "]";

            NDArray view;
            try
            {
                var nd = np.arange(prod).astype(tc).reshape(shape);
                view = slice.Length == 0 ? nd : nd[slice];
            }
            catch (Exception e)
            {
                // NumSharp couldn't parse/apply a slice numpy accepts -> potential divergence; verify numpy accepts it
                bool numpyOk;
                try { PyExec($"_chk = np.arange({prod}, dtype='{npn}').reshape({shapeStr}){npIndex}"); numpyOk = true; }
                catch { numpyOk = false; }
                if (numpyOk && parseDiv++ < 6) FIND($"fuzz: NumSharp failed slice '{slice}' on shape [{shapeStr}] that numpy accepts ({e.GetType().Name})");
                continue;
            }

            try
            {
                using (Py.GIL()) { using var a = view.ToNumpy(); Scope().Set("a", a); }
                PyExec($"base = np.arange({prod}, dtype='{npn}').reshape({shapeStr})");
                PyExec($"expected = base{npIndex}");
                bool sheq = PyB("a.shape == expected.shape");
                bool veq = PyB("np.array_equal(a, expected)");
                // strides of size-1 axes are "don't care" (numpy itself does not define them); compare
                // only the MEANINGFUL strides (axes with size > 1). A mismatch there is a real bug.
                bool steq = PyB("a.size==0 or all(a.strides[i]==expected.strides[i] for i in range(a.ndim) if a.shape[i]>1)");
                if (!(sheq && veq && steq))
                {
                    exportFails++;
                    if (exportFails <= 8) FIND($"fuzz EXPORT tc={tc} shape=[{shapeStr}] slice='{slice}': shapeEq={sheq} valEq={veq} meaningfulStrideEq={steq} | a={PyS("(a.shape,a.strides)")} exp={PyS("(expected.shape,expected.strides)")}");
                }

                // copy-import fidelity: numpy expected -> ToNDArray (C-contig copy) -> back to numpy
                using (Py.GIL())
                {
                    using var exp = Scope().Get("expected");
                    using var nd2 = NDArrayPythonInterop.ToNDArray(exp);
                    using var c = nd2.ToNumpy();
                    Scope().Set("c", c);
                }
                if (!PyB("np.array_equal(c, expected)"))
                {
                    copyFails++;
                    if (copyFails <= 8) FIND($"fuzz COPY-IMPORT tc={tc} shape=[{shapeStr}] slice='{slice}': ToNDArray(expected) != expected");
                }

                // view-import fidelity (dtype-agnostic): import expected as a zero-copy view, re-export
                // that view, and assert it equals expected. This tests that ToNDArrayView read the exact
                // layout (any wrong stride on a size>1 axis makes the re-export differ) AND aliases the
                // same memory — all compared in numpy, so it does not touch the double-only Get/SetDouble.
                {
                    using (Py.GIL())
                    {
                        using var exp = Scope().Get("expected");
                        using var v = NDArrayPythonInterop.ToNDArrayView(exp, allowReadonly: true);
                        using var b = v.ToNumpy();
                        Scope().Set("b", b);
                        Scope().Set("expected", exp);
                    }
                    if (!PyB("b.shape==expected.shape and np.array_equal(b, expected)"))
                    {
                        aliasFails++;
                        if (aliasFails <= 8) FIND($"fuzz VIEW-IMPORT tc={tc} shape=[{shapeStr}] slice='{slice}': re-exported view != expected | b={PyS("(b.shape,b.strides)")} exp={PyS("(expected.shape,expected.strides)")}");
                    }
                }
                ran++;
            }
            catch (Exception e)
            {
                if (exportFails + copyFails + aliasFails < 8) FIND($"fuzz threw tc={tc} shape=[{shapeStr}] slice='{slice}': {e.GetType().Name}: {e.Message}");
            }
            if (it % 500 == 0) Pump();
        }
        Console.WriteLine($"  fuzz: ran={ran} skipped={skipped} | exportFails={exportFails} copyFails={copyFails} aliasFails={aliasFails} parseDiv={parseDiv}");
        if (exportFails + copyFails + aliasFails + parseDiv == 0) OK($"{ran} randomized cases: export+copy+alias all match numpy");
        // release the last iteration's exports still bound in the scope, THEN check for a real leak.
        PyExec("a=b=c=expected=base=None\nimport gc\ngc.collect()");
        bool settled = WaitCounters(0, 0, 12000);
        if (!settled) FIND($"fuzz leaked after clearing scope: E={NDArrayPythonInterop.LiveExports} I={NDArrayPythonInterop.LiveImports}");
        else OK("fuzz counters settled to baseline after clearing scope (no leak across 3877 cases)");
    }

    static string RandomSlice(Random rng, int[] shape)
    {
        var toks = new string[shape.Length];
        for (int d = 0; d < shape.Length; d++)
        {
            int n = shape[d];
            switch (rng.Next(8))
            {
                case 0: toks[d] = ":"; break;
                case 1: toks[d] = $"{rng.Next(n)}:"; break;
                case 2: toks[d] = $":{rng.Next(1, n + 1)}"; break;
                case 3: { int a = rng.Next(n); int b = rng.Next(a, n + 1); toks[d] = $"{a}:{b}"; break; }
                case 4: toks[d] = "::" + (rng.Next(2) == 0 ? "2" : "3"); break;
                case 5: toks[d] = "::-1"; break;
                case 6: { int i = rng.Next(-n, n); toks[d] = $"{i}"; break; }        // single index (reduces dim)
                default: { int a = rng.Next(n); int b = rng.Next(n + 1); int s = rng.Next(2) == 0 ? -1 : -2; toks[d] = $"{a}:{b}:{s}"; break; }
            }
        }
        return string.Join(", ", toks);
    }

    // ============================================================
    //  OVERLAP: import an as_strided OVERLAPPING view (shared elements)
    // ============================================================
    static void OverlapImport()
    {
        Console.WriteLine("\n-- overlapping as_strided import --");
        try
        {
            using var scope = Py.GIL();
            // _ov[i,j] = base[i+j]; rows overlap (stride == itemsize on both axes)
            PythonEngine.RunSimpleString("import numpy as _np\n_base = _np.arange(6, dtype='f8')\n_ov = _np.lib.stride_tricks.as_strided(_base, shape=(4,3), strides=(8,8))");
            using var main = Py.Import("__main__"); using var ov = main.GetAttr("_ov");
            using var v = NDArrayPythonInterop.ToNDArrayView(ov, allowReadonly: true);
            Scope().Set("_ov2", ov);
            // read fidelity across the overlapping window
            bool ok = true;
            for (int i = 0; i < 4 && ok; i++)
                for (int j = 0; j < 3; j++)
                {
                    double got = v.GetDouble(i, j);
                    double exp = PyF($"float(_ov2[{i},{j}])");
                    if (Math.Abs(got - exp) > 1e-9) { FIND($"overlap read [{i},{j}]={got} != numpy {exp}"); ok = false; break; }
                }
            if (ok) OK("overlapping window reads all match numpy");
            // write-through to a shared cell, verify the OTHER logical index sees it (aliasing through overlap)
            if (v.Shape.IsWriteable)
            {
                v.SetDouble(42.0, 0, 1);     // base[1]
                double via10 = PyF("float(_ov2[1,0])");   // also base[1]
                if (Math.Abs(via10 - 42.0) > 1e-9) FIND($"overlap write [0,1]=42 not seen at [1,0]={via10} (shared-element aliasing broken)");
                else OK("overlap write-through visible at the aliased index");
            }
            else NOTE("overlap view non-writeable");
            v.Dispose();
        }
        catch (Exception e) { FIND($"overlap import threw {e.GetType().Name}: {e.Message}"); }
        WaitCounters(0, 0, 6000);
    }

    // ============================================================
    //  VALUEFIDELITY: complex + half bit patterns (nan/inf/subnormal), byte-exact
    // ============================================================
    static void ValueFidelity()
    {
        Console.WriteLine("\n-- value fidelity (complex / half / specials), byte-exact --");
        // complex128 export: FINITE entries byte-exact; NaN entry by class only (.NET stores negative
        // NaN 0xFFF8.. where numpy stores 0x7FF8.. — a source/dtype convention faithfully shared by the
        // zero-copy view, NOT an export defect; documented observation).
        Try("complex128 export: finite byte-exact, inf/-0 byte-exact, nan by class", () =>
        {
            var re = new System.Numerics.Complex[] {
                new(1.5,-2.5), new(double.NaN, 3.0), new(double.PositiveInfinity, double.NegativeInfinity),
                new(-0.0, 0.0), new(1e308, -1e-308) };
            var nd = np.array(re);
            using (Py.GIL()) { using var a = nd.ToNumpy(); Scope().Set("a", a); }
            PyExec("exp = np.array([1.5-2.5j, complex(np.nan,3.0), complex(np.inf,-np.inf), complex(-0.0,0.0), complex(1e308,-1e-308)], dtype='complex128')");
            bool finite = PyB("a[0].tobytes()==exp[0].tobytes()") && PyB("a[2].tobytes()==exp[2].tobytes()")
                       && PyB("a[3].tobytes()==exp[3].tobytes()") && PyB("a[4].tobytes()==exp[4].tobytes()");
            bool nanClass = PyB("np.isnan(a[1].real) and a[1].imag==3.0");
            bool netNanConvention = PyB("a[1].real.view(np.uint64)==exp[1].real.view(np.uint64)") == false;
            if (netNanConvention) NOTE("complex128 NaN exported with .NET convention (0xFFF8..) != numpy (0x7FF8..) — faithful byte-share of NumSharp's source bits");
            return PyB("a.dtype==np.complex128") && finite && nanClass;
        });
        // complex64 import widen: VALUES must match (widened to complex128)
        Try("complex64 import widen values exact", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_c8 = _np.array([1.5-2.5j, _np.complex64(complex(_np.nan,1.0))], dtype='complex64')");
            using var main = Py.Import("__main__"); using var c8 = main.GetAttr("_c8");
            using var nd = NDArrayPythonInterop.ToNDArray(c8);
            using var back = nd.ToNumpy();      // complex128
            Scope().Set("b", back);
            return PyB("b.dtype==np.complex128") && PyB("b[0]==(1.5-2.5j)") && PyB("np.isnan(b[1].real)");
        });
        // half specials byte-exact both ways
        Try("half export byte-exact (nan/inf/subnormal/-0)", () =>
        {
            var hs = new Half[] { (Half)1.5, Half.PositiveInfinity, Half.NegativeInfinity, Half.NaN,
                                  (Half)(-0.0), Half.Epsilon, Half.MaxValue, Half.MinValue };
            var nd = np.array(hs);
            using (Py.GIL()) { using var a = nd.ToNumpy(); Scope().Set("a", a); }
            // compare against numpy-constructed half array by BITS (nan payload may differ, so compare finite subset + classes)
            PyExec("exp = np.array([1.5, np.inf, -np.inf, np.nan, -0.0, np.float16(2**-24), np.float16(65504), np.float16(-65504)], dtype='float16')");
            // finite entries byte-exact; nan entry: both are nan; inf/-inf/-0 byte-exact
            return PyB("a.dtype==np.float16")
                && PyB("a[0].tobytes()==exp[0].tobytes() and a[1].tobytes()==exp[1].tobytes() and a[2].tobytes()==exp[2].tobytes()")
                && PyB("np.isnan(a[3])")
                && PyB("a[4].tobytes()==exp[4].tobytes() and a[5].tobytes()==exp[5].tobytes() and a[6].tobytes()==exp[6].tobytes() and a[7].tobytes()==exp[7].tobytes()");
        });
        // half import view value fidelity for specials
        Try("half import view specials exact", () =>
        {
            using var scope = Py.GIL();
            PythonEngine.RunSimpleString("import numpy as _np\n_h = _np.array([_np.inf, -_np.inf, 2**-24, 65504], dtype='float16')");
            using var main = Py.Import("__main__"); using var h = main.GetAttr("_h");
            using var v = NDArrayPythonInterop.ToNDArrayView(h, allowReadonly: true);
            return double.IsPositiveInfinity((double)(Half)v.GetValue(0)) && double.IsNegativeInfinity((double)(Half)v.GetValue(1));
        });
        WaitCounters(0, 0, 6000);
    }

    // ============================================================
    //  RACECODEC: concurrent RegisterCodec + convert (idempotence + no crash)
    // ============================================================
    static void RaceCodec()
    {
        Console.WriteLine("\n-- concurrent RegisterCodec + convert --");
        int trueCount = 0, errors = 0;
        int T = 12;
        var threads = new List<Thread>();
        var start = new ManualResetEventSlim(false);
        for (int t = 0; t < T; t++)
        {
            var th = new Thread(() =>
            {
                start.Wait();
                try
                {
                    if (NDArrayPythonInterop.RegisterCodec()) Interlocked.Increment(ref trueCount);
                    for (int i = 0; i < 50; i++)
                    {
                        var nd = np.arange(20).astype(NPTypeCode.Double);
                        using (Py.GIL()) { using var a = nd.ToNumpy(); }
                        nd.Dispose();
                    }
                }
                catch (Exception e) { Interlocked.Increment(ref errors); if (errors < 4) Console.WriteLine($"[race] {e.GetType().Name}: {e.Message}"); }
            });
            threads.Add(th); th.Start();
        }
        start.Set();
        foreach (var th in threads) th.Join();
        // Note: this scenario's Main already called RegisterCodec() once at startup, so racing threads should all get false.
        Console.WriteLine($"  RegisterCodec returned true {trueCount} times across {T} threads (expect 0 — startup already registered), errors={errors}");
        if (errors > 0) FIND($"{errors} errors racing RegisterCodec/convert");
        else if (trueCount > 0) FIND($"RegisterCodec returned true {trueCount}x despite startup registration (idempotence race)");
        else OK("concurrent RegisterCodec idempotent, no errors");
        WaitCounters(0, 0, 8000);
    }

    // ============================================================
    //  GETSET: is SetDouble/GetDouble converting or reinterpreting? (native vs imported view)
    // ============================================================
    static void GetSetSemantics()
    {
        Console.WriteLine("\n-- SetDouble/GetDouble semantics: native NumSharp int16 --");
        var nd = np.arange(5).astype(NPTypeCode.Int16);
        nd.SetDouble(123.0, 0);
        Console.WriteLine($"  native int16: after SetDouble(123) -> GetInt16={nd.GetInt16(0)}  GetDouble={nd.GetDouble(0)}");
        if (nd.GetInt16(0) == 123) NOTE("SetDouble CONVERTS on native array (123 stored)");
        else NOTE($"SetDouble does NOT convert on native array (GetInt16={nd.GetInt16(0)}) -> harness must use typed setters");

        Console.WriteLine("-- same on an IMPORTED int16 view --");
        using (Py.GIL())
        {
            PythonEngine.RunSimpleString("import numpy as _np\n_gi = _np.arange(5, dtype='int16')");
            using var main = Py.Import("__main__"); using var gi = main.GetAttr("_gi");
            using var v = NDArrayPythonInterop.ToNDArrayView(gi);
            v.SetDouble(123.0, 0);
            Scope().Set("_gi2", gi);
            long pySees = PyL("int(_gi2[0])");
            Console.WriteLine($"  imported int16 view: after SetDouble(123) -> numpy sees {pySees}  NumSharp GetInt16={v.GetInt16(0)}");
            NOTE(pySees == nd.GetInt16(0)
                ? "imported view matches native SetDouble semantics (consistent — not interop-specific)"
                : "imported view DIVERGES from native SetDouble semantics (interop-specific!)");
        }
        WaitCounters(0, 0, 6000);
    }

    // ============================================================
    //  CODECFUZZ: encode NDArray -> python compute -> decode As<NDArray>, vs numpy-native
    // ============================================================
    static void CodecFuzz()
    {
        Console.WriteLine("\n-- codec compute round-trip fuzz (encode->matmul/elementwise->decode) --");
        var rng = new Random(4242);
        int N = 800, fails = 0, ran = 0;
        for (int it = 0; it < N; it++)
        {
            int m = rng.Next(1, 6), k = rng.Next(1, 6), n = rng.Next(1, 6);
            bool useFloat = rng.Next(2) == 0;
            NPTypeCode tc = useFloat ? NPTypeCode.Double : NPTypeCode.Int64;
            string npn = useFloat ? "float64" : "int64";
            try
            {
                var A = np.arange(m * k).astype(tc).reshape(m, k);
                var B = np.arange(k * n).astype(tc).reshape(k, n);
                bool transposeA = rng.Next(3) == 0;
                NDArray Ause = transposeA && m == k ? A.T : A;   // sometimes feed a non-contiguous operand
                using (Py.GIL())
                {
                    using var scope = Py.CreateScope();
                    scope.Exec("import numpy as np");
                    scope.Set("a", Ause); scope.Set("b", B);       // NDArray auto-encoded
                    int op = rng.Next(3);
                    string expr = op == 0 ? "np.matmul(a, b)" : op == 1 ? "a.astype('float64').sum()" : "(a + a)";
                    using var r = scope.Eval(expr);
                    using var nd = r.As<NDArray>();                 // decoded
                    using var backc = nd.ToNumpy();
                    scope.Set("back", backc);
                    // numpy-native reference
                    string aexpr = transposeA && m == k ? $"np.arange({m*k}, dtype='{npn}').reshape({m},{k}).T" : $"np.arange({m*k}, dtype='{npn}').reshape({m},{k})";
                    scope.Exec($"ea = {aexpr}\neb = np.arange({k*n}, dtype='{npn}').reshape({k},{n})");
                    string refExpr = op == 0 ? "np.matmul(ea, eb)" : op == 1 ? "ea.astype('float64').sum()" : "(ea + ea)";
                    scope.Exec($"ref = {refExpr}");
                    using var eq = scope.Eval("bool(np.array_equal(np.asarray(back), np.asarray(ref)))");
                    if (!eq.As<bool>())
                    {
                        fails++;
                        if (fails <= 6)
                        {
                            using var bs = scope.Eval("str((np.asarray(back).shape, np.asarray(ref).shape))");
                            FIND($"codecfuzz op={op} m={m} k={k} n={n} tc={tc} tA={transposeA}: decoded result != numpy-native {bs.As<string>()}");
                        }
                    }
                }
                ran++;
            }
            catch (Exception e) { if (fails++ < 6) FIND($"codecfuzz threw m={m} k={k} n={n} tc={tc}: {e.GetType().Name}: {e.Message}"); }
            if (it % 200 == 0) Pump();
        }
        Console.WriteLine($"  codecfuzz: ran={ran} fails={fails}");
        if (fails == 0) OK($"{ran} codec compute round-trips match numpy-native");
        bool settled = WaitCounters(0, 0, 10000);
        if (!settled) FIND($"codecfuzz leaked: E={NDArrayPythonInterop.LiveExports} I={NDArrayPythonInterop.LiveImports}");
        else OK("codecfuzz counters settled");
    }

    // ============================================================
    //  TORTURE: mixed export/import/copy/finalizer-drop across threads + aggressive GC
    // ============================================================
    static void Torture()
    {
        Console.WriteLine("\n-- concurrency + GC torture (mixed conversions, 6s) --");
        var scope = Scope();
        int T = 10; long deadline = 6000;
        int errors = 0; long ops = 0;
        var sw = Stopwatch.StartNew();
        var threads = new List<Thread>();
        for (int t = 0; t < T; t++)
        {
            int tid = t;
            var th = new Thread(() =>
            {
                var rng = new Random(1000 + tid);
                while (sw.ElapsedMilliseconds < deadline)
                {
                    try
                    {
                        int op = rng.Next(6);
                        int sz = rng.Next(1, 2000);
                        switch (op)
                        {
                            case 0: { var nd = np.arange(sz).astype(NPTypeCode.Double); using (Py.GIL()) { using var a = nd.ToNumpy(); } nd.Dispose(); break; }
                            case 1: { var nd = np.arange(sz).astype(NPTypeCode.Single); using (Py.GIL()) { using var a = nd.ToNumpyCopy(); } nd.Dispose(); break; }
                            case 2: { using (Py.GIL()) { PythonEngine.RunSimpleString($"import numpy as _np\n_tx{tid} = _np.arange({sz}, dtype='f8')"); using var main = Py.Import("__main__"); using var s = main.GetAttr($"_tx{tid}"); using var v = NDArrayPythonInterop.ToNDArrayView(s); } break; }
                            case 3: { using (Py.GIL()) { PythonEngine.RunSimpleString($"import numpy as _np\n_ty{tid} = _np.arange({sz}, dtype='i4')"); using var main = Py.Import("__main__"); using var s = main.GetAttr($"_ty{tid}"); using var nd = NDArrayPythonInterop.ToNDArray(s); } break; }
                            case 4: DropImport(tid, sz); break;   // finalizer path
                            case 5: if (rng.Next(4) == 0) { GC.Collect(); using (Py.GIL()) PythonEngine.RunSimpleString("import gc; gc.collect()"); } break;
                        }
                        Interlocked.Increment(ref ops);
                    }
                    catch (Exception e) { Interlocked.Increment(ref errors); if (errors <= 5) Console.WriteLine($"[torture {tid}] {e.GetType().Name}: {e.Message}"); }
                }
            });
            threads.Add(th); th.Start();
        }
        foreach (var th in threads) th.Join();
        Console.WriteLine($"  torture: {ops} ops across {T} threads, errors={errors}");
        if (errors > 0) FIND($"{errors} errors during concurrency/GC torture");
        else OK($"{ops} mixed conversions across {T} threads with aggressive GC — no errors/crash/hang");
        bool settled = WaitCounters(0, 0, 15000);
        if (!settled) FIND($"torture counters did not settle: E={NDArrayPythonInterop.LiveExports} I={NDArrayPythonInterop.LiveImports}");
        else OK("torture counters settled to baseline");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    static void DropImport(int tid, int sz)
    {
        using (Py.GIL())
        {
            PythonEngine.RunSimpleString($"import numpy as _np\n_tz{tid} = _np.arange({sz}, dtype='f8')");
            using var main = Py.Import("__main__"); using var s = main.GetAttr($"_tz{tid}");
            var v = NDArrayPythonInterop.ToNDArrayView(s);   // NOT disposed -> finalizer releases
        }
    }

    // ============================================================
    //  MEMVIEWFUZZ: ToMemoryView bytes vs numpy tobytes + write-through
    // ============================================================
    static void MemViewFuzz()
    {
        Console.WriteLine("\n-- ToMemoryView fidelity + write-through --");
        var scope = Scope();
        (NPTypeCode tc, string np)[] dt =
        {
            (NPTypeCode.Byte,"uint8"),(NPTypeCode.Int16,"int16"),(NPTypeCode.Int32,"int32"),
            (NPTypeCode.Int64,"int64"),(NPTypeCode.Single,"float32"),(NPTypeCode.Double,"float64"),(NPTypeCode.Half,"float16"),
        };
        int fails = 0;
        foreach (var (tc, npn) in dt)
        {
            try
            {
                int n = 12;
                var nd = np.arange(n).astype(tc);
                using (Py.GIL()) { using var mv = nd.ToMemoryView(); Scope().Set("mv", mv); }
                bool byteEq = PyB($"bytes(mv) == np.arange({n}, dtype='{npn}').tobytes()");
                if (!byteEq) { fails++; FIND($"memview {tc}: bytes != np.arange tobytes"); }
                else OK($"memview {tc}: bytes match np.arange");
            }
            catch (Exception e) { fails++; FIND($"memview {tc} threw {e.GetType().Name}: {e.Message}"); }
        }
        // offset-contiguous window (a row of a 2-D array is contiguous)
        try
        {
            var m2 = np.arange(20).astype(NPTypeCode.Double).reshape(4, 5);
            var row = m2[2];   // contiguous, offset 10
            using (Py.GIL()) { using var mv = row.ToMemoryView(); Scope().Set("mv", mv); }
            bool eq = PyB("bytes(mv) == np.arange(10,15, dtype='f8').tobytes()");
            if (!eq) { fails++; FIND("memview offset-row: bytes != expected window [10..15)"); }
            else OK("memview offset-row window correct");
            // write-through: mutate byte 0 region via numpy, check NumSharp sees
            using (Py.GIL()) Scope().Exec("import struct\nmv[0:8] = struct.pack('<d', 777.0)");
            if (Math.Abs(row.GetDouble(0) - 777.0) > 1e-9) { fails++; FIND($"memview write-through: wrote 777 via memoryview, NumSharp sees {row.GetDouble(0)}"); }
            else OK("memview write-through visible in NumSharp");
        }
        catch (Exception e) { fails++; FIND($"memview offset test threw {e.GetType().Name}: {e.Message}"); }
        if (fails == 0) OK("ToMemoryView all correct");
        WaitCounters(0, 0, 8000);
    }

    // ============================================================
    //  SOAK: longer leak run (2000 iters) for slow leaks
    // ============================================================
    static void Soak()
    {
        var scope = Scope();
        const int N = 2000, elems = 200_000;   // 1.6MB each
        Console.WriteLine($"\n-- soak: {N} mixed iterations --");
        long baseWs = Working();
        var rng = new Random(7);
        for (int i = 0; i < N; i++)
        {
            int op = rng.Next(4);
            if (op == 0) { var nd = np.arange(elems).astype(NPTypeCode.Double); using (Py.GIL()) { using var a = nd.ToNumpy(); } nd.Dispose(); }
            else if (op == 1) { var nd = np.arange(elems).astype(NPTypeCode.Single); using (Py.GIL()) { using var a = nd.ToNumpyCopy(); } nd.Dispose(); }
            else if (op == 2) { using (Py.GIL()) { PythonEngine.RunSimpleString($"import numpy as _np\n_s = _np.arange({elems}, dtype='f8')"); using var main = Py.Import("__main__"); using var s = main.GetAttr("_s"); using var v = NDArrayPythonInterop.ToNDArrayView(s); v.SetDouble(1, 0); main.SetAttr("_s", 0.ToPython()); } }
            else { using (Py.GIL()) { PythonEngine.RunSimpleString($"import numpy as _np\n_s = _np.arange({elems}, dtype='f8')"); using var main = Py.Import("__main__"); using var s = main.GetAttr("_s"); using var nd = NDArrayPythonInterop.ToNDArray(s); main.SetAttr("_s", 0.ToPython()); } }
            if (i % 200 == 0) { Pump(); if (i % 800 == 0) Console.WriteLine($"  iter {i}: WS={Working()/1024/1024}MB (+{(Working()-baseWs)/1024/1024}) E={NDArrayPythonInterop.LiveExports} I={NDArrayPythonInterop.LiveImports}"); }
        }
        Pump(); WaitCounters(0, 0, 12000);
        long grow = Working() - baseWs;
        Console.WriteLine($"  soak done: net WS growth {grow/1024/1024}MB, E={NDArrayPythonInterop.LiveExports} I={NDArrayPythonInterop.LiveImports}");
        if (grow > 300L * 1024 * 1024) FIND($"soak WS grew {grow/1024/1024}MB over {N} iters (possible slow leak)");
        else OK($"soak WS growth bounded ({grow/1024/1024}MB over {N} iters)");
    }

    // ============================================================
    //  BYTEIDENT: same np.* op in numpy vs NumSharp, compare RAW BYTES
    //   - identical INPUT bytes (finite literals, or numpy specials imported byte-exact)
    //   - identical op in each engine
    //   - compare output dtype + tobytes() (hex on mismatch)
    // ============================================================
    static int _biSame, _biDiff, _biConv;
    static void ByteIdent()
    {
        _biSame = _biDiff = _biConv = 0;
        var scope = Scope();

        Console.WriteLine("\n== FAMILY A: produce specials from FINITE inputs (inputs trivially byte-identical) ==");
        // float64
        Cmp("f8 0/0", () => np.array(new[] { 0.0 }) / np.array(new[] { 0.0 }), "np.array([0.0])/np.array([0.0])");
        Cmp("f8 1/0", () => np.array(new[] { 1.0 }) / np.array(new[] { 0.0 }), "np.array([1.0])/np.array([0.0])");
        Cmp("f8 -1/0", () => np.array(new[] { -1.0 }) / np.array(new[] { 0.0 }), "np.array([-1.0])/np.array([0.0])");
        Cmp("f8 sqrt(-1)", () => np.sqrt(np.array(new[] { -1.0 })), "np.sqrt(np.array([-1.0]))");
        Cmp("f8 log(-1)", () => np.log(np.array(new[] { -1.0 })), "np.log(np.array([-1.0]))");
        Cmp("f8 log(0)", () => np.log(np.array(new[] { 0.0 })), "np.log(np.array([0.0]))");
        Cmp("f8 inf-inf", () => (np.array(new[] { 1.0 }) / np.array(new[] { 0.0 })) - (np.array(new[] { 1.0 }) / np.array(new[] { 0.0 })), "np.array([np.inf])-np.array([np.inf])");
        Cmp("f8 0*inf", () => np.array(new[] { 0.0 }) * (np.array(new[] { 1.0 }) / np.array(new[] { 0.0 })), "np.array([0.0])*np.array([np.inf])");
        Cmp("f8 exp(1000)", () => np.exp(np.array(new[] { 1000.0 })), "np.exp(np.array([1000.0]))");
        Cmp("f8 exp(-1000)", () => np.exp(np.array(new[] { -1000.0 })), "np.exp(np.array([-1000.0]))");
        // float32
        Cmp("f4 0/0", () => np.array(new[] { 0f }) / np.array(new[] { 0f }), "np.array([0.0],dtype='f4')/np.array([0.0],dtype='f4')");
        Cmp("f4 sqrt(-1)", () => np.sqrt(np.array(new[] { -1f })), "np.sqrt(np.array([-1.0],dtype='f4'))");
        Cmp("f4 log(-1)", () => np.log(np.array(new[] { -1f })), "np.log(np.array([-1.0],dtype='f4'))");
        Cmp("f4 exp(1000)", () => np.exp(np.array(new[] { 1000f })), "np.exp(np.array([1000.0],dtype='f4'))");
        // float16
        Cmp("f2 sqrt(-1)", () => np.sqrt(np.array(new[] { (Half)(-1.0) })), "np.sqrt(np.array([-1.0],dtype='f2'))");
        Cmp("f2 0/0", () => np.array(new[] { (Half)0.0 }) / np.array(new[] { (Half)0.0 }), "np.array([0.0],dtype='f2')/np.array([0.0],dtype='f2')");

        Console.WriteLine("\n== FAMILY B: transform GIVEN specials (input bytes imported from numpy => identical) ==");
        // f8/f4/f2 specials vector: [1,-1,0,-0,inf,-inf,nan,2.5,-0.5,subnormal,max]
        foreach (var (dt, npdt) in new[] { ("f8", "float64"), ("f4", "float32"), ("f2", "float16") })
        {
            string subnormal = dt == "f8" ? "5e-324" : dt == "f4" ? "1e-45" : "6e-8";
            string maxv = dt == "f8" ? "1.7976931348623157e308" : dt == "f4" ? "3.4e38" : "65504";
            string vec(string ns) => $"{ns}.array([1.0,-1.0,0.0,-0.0,{ns}.inf,-{ns}.inf,{ns}.nan,2.5,-0.5,{subnormal},{maxv}], dtype='{dt}')";
            string npVec = vec("np"), importVec = vec("_np");

            CmpB($"{dt} sqrt(x)", importVec, n => np.sqrt(n), $"np.sqrt({npVec})");
            CmpB($"{dt} log(x)", importVec, n => np.log(n), $"np.log({npVec})");
            CmpB($"{dt} exp(x)", importVec, n => np.exp(n), $"np.exp({npVec})");
            CmpB($"{dt} sin(x)", importVec, n => np.sin(n), $"np.sin({npVec})");
            CmpB($"{dt} cos(x)", importVec, n => np.cos(n), $"np.cos({npVec})");
            CmpB($"{dt} abs(x)", importVec, n => np.abs(n), $"np.abs({npVec})");
            CmpB($"{dt} negative(x)", importVec, n => np.negative(n), $"np.negative({npVec})");
            CmpB($"{dt} square(x)", importVec, n => np.square(n), $"np.square({npVec})");
            CmpB($"{dt} sign(x)", importVec, n => np.sign(n), $"np.sign({npVec})");
            CmpB($"{dt} reciprocal(x)", importVec, n => np.reciprocal(n), $"np.reciprocal({npVec})");
            CmpB($"{dt} x+x", importVec, n => n + n, $"({npVec})+({npVec})");
            CmpB($"{dt} x-x", importVec, n => n - n, $"({npVec})-({npVec})");
            CmpB($"{dt} x*x", importVec, n => n * n, $"({npVec})*({npVec})");
            CmpB($"{dt} x/x", importVec, n => n / n, $"({npVec})/({npVec})");
            // reductions
            CmpB($"{dt} sum(x)", importVec, n => np.sum(n), $"np.sum({npVec})");
            CmpB($"{dt} min(x)", importVec, n => np.min(n), $"np.min({npVec})");
            CmpB($"{dt} max(x)", importVec, n => np.max(n), $"np.max({npVec})");
        }

        Console.WriteLine("\n== complex128 specials ==");
        {
            string cvec(string ns) => $"{ns}.array([1+2j, complex({ns}.nan,1.0), complex(1.0,{ns}.nan), complex({ns}.inf,-{ns}.inf), 0+0j, complex(-0.0,0.0)], dtype='complex128')";
            CmpB("c16 x+x", cvec("_np"), n => n + n, $"({cvec("np")})+({cvec("np")})");
            CmpB("c16 x*x", cvec("_np"), n => n * n, $"({cvec("np")})*({cvec("np")})");
            CmpB("c16 abs(x)", cvec("_np"), n => np.abs(n), $"np.abs({cvec("np")})");
            CmpB("c16 sqrt(x)", cvec("_np"), n => np.sqrt(n), $"np.sqrt({cvec("np")})");
        }

        Console.WriteLine("\n== CONSTRUCTION convention (C# double.NaN vs numpy np.nan) ==");
        Cmp("f8 full(NaN) construct", () => np.full(new Shape(3), double.NaN, NPTypeCode.Double), "np.full(3, np.nan)");
        Cmp("f8 array([NaN]) construct", () => np.array(new[] { double.NaN }), "np.array([np.nan])");

        Console.WriteLine($"\n  byteident: identical={_biSame}  divergent={_biDiff}  (conversion-mismatch={_biConv})");
        if (_biDiff == 0) OK($"all {_biSame} op comparisons byte-identical to numpy");
        else FIND($"{_biDiff} op comparisons DIVERGE from numpy in raw bytes on supported dtypes");
        WaitCounters(0, 0, 8000);
    }

    // ============================================================
    //  NANSWEEP: map the full NaN-bit divergence surface (minimal inputs)
    // ============================================================
    static void NanSweep()
    {
        _biSame = _biDiff = _biConv = 0;
        var scope = Scope();
        foreach (var dt in new[] { "f8", "f4", "f2" })
        {
            Console.WriteLine($"\n== {dt} ==");
            string pos = $"{{0}}.array([{{0}}.nan], dtype='{dt}')";
            string infpair = $"{{0}}.array([{{0}}.inf, -{{0}}.inf], dtype='{dt}')";
            string mix = $"{{0}}.array([1.0, {{0}}.nan, 2.0], dtype='{dt}')";
            string zero = $"{{0}}.array([0.0], dtype='{dt}')";
            string I(string t) => string.Format(t, "_np");
            string N(string t) => string.Format(t, "np");

            CmpB($"{dt} sign(nan)", I(pos), n => np.sign(n), $"np.sign({N(pos)})");
            CmpB($"{dt} negative(nan)", I(pos), n => np.negative(n), $"np.negative({N(pos)})");
            CmpB($"{dt} abs(nan)", I(pos), n => np.abs(n), $"np.abs({N(pos)})");
            CmpB($"{dt} sqrt(nan)", I(pos), n => np.sqrt(n), $"np.sqrt({N(pos)})");
            CmpB($"{dt} square(nan)", I(pos), n => np.square(n), $"np.square({N(pos)})");
            CmpB($"{dt} reciprocal(nan)", I(pos), n => np.reciprocal(n), $"np.reciprocal({N(pos)})");
            CmpB($"{dt} nan+1", I(pos), n => n + np.array(new[] { 1.0 }).astype(n.typecode), $"({N(pos)})+np.array([1.0],dtype='{dt}')");
            CmpB($"{dt} sum(inf,-inf)", I(infpair), n => np.sum(n), $"np.sum({N(infpair)})");
            CmpB($"{dt} sum([1,nan,2])", I(mix), n => np.sum(n), $"np.sum({N(mix)})");
            CmpB($"{dt} mean([1,nan,2])", I(mix), n => np.mean(n), $"np.mean({N(mix)})");
            CmpB($"{dt} prod([1,nan,2])", I(mix), n => np.prod(n), $"np.prod({N(mix)})");
            CmpB($"{dt} std([1,nan,2])", I(mix), n => np.std(n), $"np.std({N(mix)})");
            CmpB($"{dt} var([1,nan,2])", I(mix), n => np.var(n), $"np.var({N(mix)})");
            CmpB($"{dt} min([1,nan,2])", I(mix), n => np.amin(n), $"np.amin({N(mix)})");
            CmpB($"{dt} max([1,nan,2])", I(mix), n => np.amax(n), $"np.amax({N(mix)})");
            CmpB($"{dt} maximum(nan,1)", I(pos), n => np.maximum(n, np.array(new[] { 1.0 }).astype(n.typecode)), $"np.maximum({N(pos)}, np.array([1.0],dtype='{dt}'))");
            CmpB($"{dt} 0/0", I(zero), n => n / n, $"({N(zero)})/({N(zero)})");
        }
        Console.WriteLine("\n== complex128 ==");
        string cnan1 = "{0}.array([complex(1.0,{0}.nan)], dtype='complex128')";
        string cnan2 = "{0}.array([complex({0}.nan,1.0)], dtype='complex128')";
        CmpB("c16 sqrt(1+nanj)", string.Format(cnan1, "_np"), n => np.sqrt(n), $"np.sqrt({string.Format(cnan1, "np")})");
        CmpB("c16 sqrt(nan+1j)", string.Format(cnan2, "_np"), n => np.sqrt(n), $"np.sqrt({string.Format(cnan2, "np")})");
        CmpB("c16 log(1+nanj)", string.Format(cnan1, "_np"), n => np.log(n), $"np.log({string.Format(cnan1, "np")})");
        CmpB("c16 exp(1+nanj)", string.Format(cnan1, "_np"), n => np.exp(n), $"np.exp({string.Format(cnan1, "np")})");
        CmpB("c16 reciprocal(1+nanj)", string.Format(cnan1, "_np"), n => np.reciprocal(n), $"np.reciprocal({string.Format(cnan1, "np")})");

        Console.WriteLine($"\n  nansweep: identical={_biSame}  divergent={_biDiff}");
        if (_biDiff == 0) OK($"all {_biSame} nan comparisons byte-identical");
        else FIND($"{_biDiff} nan-producing ops diverge in raw bytes");
        WaitCounters(0, 0, 8000);
    }

    // Family A: independent finite inputs -> compare result bytes
    static void Cmp(string label, Func<NDArray> ns, string npExpr)
    {
        try
        {
            NDArray r = ns();
            using (Py.GIL()) { using var a = r.ToNumpy(); Scope().Set("nsr", a); }
            PyExec($"npr = np.asarray({npExpr})");
            bool dtEq = PyB("nsr.dtype == npr.dtype");
            bool byteEq = PyB("nsr.tobytes() == npr.tobytes()");
            if (dtEq && byteEq) { _biSame++; OK($"{label}: byte-identical [{PyS("nsr.dtype")}]"); }
            else
            {
                if (!dtEq) _biConv++;
                _biDiff++;
                FIND($"{label}: NS={PyS("nsr.dtype")}:{PyS("nsr.tobytes().hex()")} NP={PyS("npr.dtype")}:{PyS("npr.tobytes().hex()")}");
            }
        }
        catch (Exception e) { _biDiff++; FIND($"{label} threw {e.GetType().Name}: {e.Message}"); }
    }

    // Family B: import numpy-built input (byte-exact), run op in NumSharp, compare to numpy-native
    static void CmpB(string label, string importExpr, Func<NDArray, NDArray> nsOp, string npExpr)
    {
        try
        {
            NDArray xns;
            using (Py.GIL())
            {
                PythonEngine.RunSimpleString($"import numpy as _np\n_bx = {importExpr}");
                using var main = Py.Import("__main__"); using var bx = main.GetAttr("_bx");
                xns = NDArrayPythonInterop.ToNDArray(bx);   // byte-exact copy of the input
            }
            NDArray r = nsOp(xns);
            using (Py.GIL()) { using var a = r.ToNumpy(); Scope().Set("nsr", a); }
            PyExec($"npr = np.asarray({npExpr})");
            bool dtEq = PyB("nsr.dtype == npr.dtype");
            bool byteEq = PyB("nsr.shape==npr.shape and nsr.tobytes() == npr.tobytes()");
            if (dtEq && byteEq) { _biSame++; OK($"{label}: byte-identical [{PyS("nsr.dtype")}]"); }
            else
            {
                if (!dtEq) _biConv++;
                _biDiff++;
                // show only first differing element index for readability
                FIND($"{label}: NS[{PyS("nsr.dtype")}]={PyS("nsr.tobytes().hex()")} | NP[{PyS("npr.dtype")}]={PyS("npr.tobytes().hex()")}");
            }
            xns.Dispose();
        }
        catch (Exception e) { _biDiff++; FIND($"{label} threw {e.GetType().Name}: {e.Message}"); }
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
