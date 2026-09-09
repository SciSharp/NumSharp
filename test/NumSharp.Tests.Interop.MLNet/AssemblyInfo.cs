using Microsoft.VisualStudio.TestTools.UnitTesting;

// The suite asserts on the process-global NDArrayMLNetInterop.LiveExports lifetime counter (every test
// doubles as a no-leak / no-premature-free gate), so tests must run sequentially.
[assembly: DoNotParallelize]
