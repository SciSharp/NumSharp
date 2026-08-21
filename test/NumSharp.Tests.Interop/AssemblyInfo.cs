using Microsoft.VisualStudio.TestTools.UnitTesting;

// The interop tests share one embedded Python engine (CPython cannot re-initialize numpy in a
// process) and assert on the process-global LiveExports/LiveImports lifetime counters, so they
// must run sequentially.
[assembly: DoNotParallelize]
