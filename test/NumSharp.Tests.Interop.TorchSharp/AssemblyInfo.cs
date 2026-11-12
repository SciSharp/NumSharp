using Microsoft.VisualStudio.TestTools.UnitTesting;

// The suite asserts on the process-global NDArrayTorchInterop.LiveExports / LiveImports counters and on
// SizeBucketedBufferPool's foreign bookings (every test doubles as a no-leak / no-premature-free gate), so tests run
// one at a time.
[assembly: DoNotParallelize]
