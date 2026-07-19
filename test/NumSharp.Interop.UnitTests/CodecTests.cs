using System;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Auto-marshaling through pythonnet's conversion pipeline. Registration is process-global and
    ///     sticky for the engine session, so tests assert behavior, not registration order.
    /// </summary>
    [TestClass]
    public class CodecTests : InteropTestBase
    {
        /// <summary>Register once for the whole run; safe no matter which test class runs first.</summary>
        internal static void EnsureCodec()
        {
            NDArrayPythonInterop.RegisterCodec();
        }

        [TestMethod]
        public void Register_IsIdempotentPerSession()
        {
            EnsureCodec();
            NDArrayPythonInterop.RegisterCodec().Should().BeFalse("second registration in one engine session must be a no-op");
        }

        [TestMethod]
        public void AutoEncode_OnScopeSet_ProducesASharedNumpyView()
        {
            EnsureCodec();
            var nd = np.arange(4).astype(NPTypeCode.Double);

            using (Gil())
                Scope.Set("auto", nd);   // no explicit conversion anywhere

            PyStr("type(auto).__name__").Should().Be("ndarray");
            PyExec("auto[1] = 88.5");
            ReadAt<double>(nd, 1).Should().BeApproximately(88.5, 1e-12, "the codec encodes as a zero-copy view by default");
        }

        [TestMethod]
        public void AutoEncode_OnCallArguments()
        {
            EnsureCodec();
            PyExec("def fsum(a):\n    return float(a.sum())");

            var nd = np.arange(5).astype(NPTypeCode.Double);
            double sum;
            using (Gil())
            {
                dynamic fsum = Scope.Get("fsum");
                sum = (double)fsum(nd);   // NDArray argument encoded by the codec on the way in
            }

            sum.Should().BeApproximately(10.0, 1e-9);
        }

        [TestMethod]
        public void Decode_NdarrayToNDArray_DefaultsToView()
        {
            // The registered codec's DecodeMode default is Auto: a contiguous numpy source has a
            // zero-copy NumSharp representation, so As<NDArray>() shares its memory (view-first).
            EnsureCodec();
            PyExec("mm = np.arange(6, dtype='f8').reshape(2, 3)");

            NDArray dec;
            using (Gil())
            {
                using var mmPy = Scope.Get("mm");
                dec = mmPy.As<NDArray>();
            }

            dec.typecode.Should().Be(NPTypeCode.Double);
            dec.shape[0].Should().Be(2);
            dec.shape[1].Should().Be(3);
            ReadAt<double>(dec, 1, 2).Should().BeApproximately(5.0, 1e-12);

            WriteAt(dec, -1.0, 0, 0);
            PyFloat("float(mm[0, 0])").Should().BeApproximately(-1.0, 1e-12,
                "Auto decode of a contiguous source is a zero-copy view — the write reaches Python");

            dec.Dispose();   // release the import lease deterministically (base-class leak gate)
        }

        [TestMethod]
        public void Decode_CoversSubclassesAndBufferBuiltins()
        {
            EnsureCodec();

            // Under the Auto default these decode as views (leases), so dispose them deterministically.
            using (Gil())
            {
                using (var mx = Scope.Eval("np.matrix('1 2; 3 4').astype('i8')"))
                using (var dmx = mx.As<NDArray>())
                {
                    dmx.shape[0].Should().Be(2);
                    dmx.shape[1].Should().Be(2);
                    ReadAt<long>(dmx, 1, 1).Should().Be(4, "numpy.matrix decodes via the __mro__ walk");
                }

                using (var mv = Scope.Eval("memoryview(b'ab')"))
                using (var dm = mv.As<NDArray>())
                {
                    dm.typecode.Should().Be(NPTypeCode.Byte);
                    ReadAt<byte>(dm, 1).Should().Be(98);
                }

                using (var bs = Scope.Eval("b'xyz'"))
                using (var db = bs.As<NDArray>())
                {
                    db.typecode.Should().Be(NPTypeCode.Byte);
                    ReadAt<byte>(db, 0).Should().Be((byte)'x');
                }
            }
        }

        [TestMethod]
        public void Encode_Decimal_FallsBackToClrWrapping_InsteadOfFailing()
        {
            EnsureCodec();
            var dec = np.arange(3).astype(NPTypeCode.Decimal);

            using (Gil())
                Scope.Set("dauto", dec);

            PyStr("type(dauto).__name__").Should().Be("NDArray",
                "no numpy dtype exists for decimal, so pythonnet's CLR-object wrapping must take over");
        }
    }
}
