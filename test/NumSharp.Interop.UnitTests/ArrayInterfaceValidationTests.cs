using Microsoft.VisualStudio.TestTools.UnitTesting;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace NumSharp.Interop.UnitTests
{
    /// <summary>
    ///     Regression gate for the <c>__array_interface__</c> import-validation parity fix.
    ///
    ///     <para><b>The gap:</b> <see cref="NDArrayPythonInterop.ToNDArrayView(PyObject, bool, bool?)"/>
    ///     read <c>__array_interface__['data'][0]</c>, <c>['shape']</c> and <c>['strides']</c> from an
    ///     arbitrary Python object and built a zero-copy NumSharp view over them WITHOUT the validation
    ///     numpy performs. A hostile or buggy exporter could therefore make the interop build a view over
    ///     a NULL/garbage address, a negative extent, or with a strides tuple whose length disagreed with
    ///     the shape — the memory-unsafe opposite of this package's zero-copy-safety contract. numpy
    ///     rejects every one of these with a <c>ValueError</c>; the interop now rejects them too, with a
    ///     <see cref="System.NotSupportedException"/> (the package's house type for "cannot view this").</para>
    ///
    ///     <para>Each malformed case is paired with the numpy message it mirrors:
    ///     null data -&gt; "data is NULL but array contains data"; negative shape -&gt; "negative
    ///     dimensions are not allowed"; strides/shape length mismatch -&gt; "mismatch in length of
    ///     strides and shape". The control test proves a genuinely valid strided interface still imports
    ///     zero-copy, so the guards did not over-reject.</para>
    /// </summary>
    [TestClass]
    public class ArrayInterfaceValidationTests : InteropTestBase
    {
        private PyObject BuildInterfaceObject(string name, string aiLiteral)
        {
            // A plain Python object carrying only __array_interface__ (no buffer protocol), so the
            // import path routes straight into ViewViaArrayInterface.
            PyExec($"class {name}:\n pass\n{name}_i = {name}()\n{name}.__array_interface__ = {aiLiteral}");
            using (Gil()) return Scope.Get($"{name}_i");
        }

        private void AssertRejected(PyObject obj, string messageFragment)
        {
            try
            {
                var ex = Assert.ThrowsException<System.NotSupportedException>(
                    () => NDArrayPythonInterop.ToNDArrayView(obj, allowReadonly: true));
                ex.Message.Should().Contain(messageFragment);
            }
            finally
            {
                using (Gil()) obj.Dispose();
            }
        }

        [TestMethod]
        public void NullDataPointer_IsRejected_NotAViewOverAddressZero()
        {
            var b = BuildInterfaceObject("_AiNull",
                "{'shape': (3,), 'typestr': '<f8', 'data': (0, False), 'version': 3}");
            AssertRejected(b, "NULL");
        }

        [TestMethod]
        public void NegativeDimension_IsRejected()
        {
            var b = BuildInterfaceObject("_AiNeg",
                "{'shape': (-2,), 'typestr': '<f8', 'data': (999999, False), 'version': 3}");
            AssertRejected(b, "negative");
        }

        [TestMethod]
        public void StridesLongerThanShape_IsRejected()
        {
            var b = BuildInterfaceObject("_AiStrLong",
                "{'shape': (2, 2), 'typestr': '<f8', 'data': (999999, False), 'strides': (8, 8, 8), 'version': 3}");
            AssertRejected(b, "mismatch in length");
        }

        [TestMethod]
        public void StridesShorterThanShape_IsRejected_Cleanly()
        {
            // Previously threw a raw IndexOutOfRangeException from the stride-normalization loop.
            var b = BuildInterfaceObject("_AiStrShort",
                "{'shape': (2, 2), 'typestr': '<f8', 'data': (999999, False), 'strides': (8,), 'version': 3}");
            AssertRejected(b, "mismatch in length");
        }

        [TestMethod]
        public void ValidStridedInterface_StillImportsZeroCopy_AndAliases()
        {
            // Control: a real numpy transpose exposes a legitimate strided __array_interface__.
            // The guards must NOT reject it, and the shared-memory contract must hold.
            using (Gil())
                PythonEngine.RunSimpleString("import numpy as _np\n_aiok = _np.arange(6, dtype='f8').reshape(2, 3).T");

            PyObject src;
            using (Gil()) { using var main = Py.Import("__main__"); src = main.GetAttr("_aiok"); }

            try
            {
                using NDArray view = NDArrayPythonInterop.ToNDArrayView(src);   // must succeed (writeable)
                view.shape.Should().Equal(3L, 2L);
                view.SetDouble(42.0, 2, 1);   // write through the view

                using (Gil())
                {
                    Scope.Set("_aiok2", src);
                    using var r = Scope.Eval("float(_aiok2[2, 1])");   // _aiok is the transpose of the base
                    r.As<double>().Should().Be(42.0, "the zero-copy view must alias the numpy array");
                }
            }
            finally
            {
                using (Gil()) src.Dispose();
            }
        }
    }
}
